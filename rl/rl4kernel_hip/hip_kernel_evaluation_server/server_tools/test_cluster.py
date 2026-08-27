#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
HIP Kernel Evaluation Cluster Test Script

This script verifies that the multi-node cluster is properly configured and running.
It tests:
1. Master server health
2. Worker node connectivity
3. End-to-end kernel evaluation

Usage:
    python server_tools/test_cluster.py --master http://localhost:8080
    python server_tools/test_cluster.py --master http://10.254.6.40:8080 --verbose
"""

import argparse
import sys
import time
import requests
from typing import Dict, List, Optional


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, success: bool, details: str = ""):
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


def test_master_health(master_url: str, verbose: bool = False) -> bool:
    """Test master server health endpoint"""
    print_header("Testing Master Server Health")
    
    try:
        response = requests.get(f"{master_url}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print_result("Master health check", True)
        if verbose:
            print(f"         Role: {data.get('role', 'unknown')}")
            print(f"         Total slots: {data.get('total_slots', 0)}")
            print(f"         Local slots: {data.get('local_slots', 0)}")
            print(f"         Remote slots: {data.get('remote_slots', 0)}")
        
        return True
    except requests.exceptions.ConnectionError:
        print_result("Master health check", False, "Connection refused")
        return False
    except Exception as e:
        print_result("Master health check", False, str(e))
        return False


def test_cluster_status(master_url: str, verbose: bool = False) -> Dict:
    """Test cluster status endpoint"""
    print_header("Testing Cluster Status")
    
    try:
        response = requests.get(f"{master_url}/cluster/status", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        total_nodes = data.get("total_nodes", 0)
        healthy_nodes = data.get("healthy_nodes", 0)
        total_gpus = data.get("total_gpus", 0)
        ready = data.get("ready", False)
        
        print_result("Cluster status check", ready)
        print(f"         Total nodes: {healthy_nodes}/{total_nodes}")
        print(f"         Total GPUs: {total_gpus}")
        
        if verbose and "workers" in data:
            print("\n  Worker Status:")
            for worker, healthy in data["workers"].items():
                status = "✓" if healthy else "✗"
                print(f"         {status} {worker}")
        
        return data
    except Exception as e:
        print_result("Cluster status check", False, str(e))
        return {}


def test_simple_evaluation(master_url: str, verbose: bool = False) -> bool:
    """Test a simple kernel evaluation"""
    print_header("Testing Simple Kernel Evaluation")
    
    # Simple HIP kernel for testing
    test_hip_code = '''
#include <hip/hip_runtime.h>

extern "C" __global__ void test_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
'''
    
    test_pytorch_code = '''
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, a, b):
        return a + b

def get_inputs():
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    return [a, b]

def get_init_inputs():
    return []
'''
    
    test_request = {
        "requests": [
            {
                "kernel_name": "test_cluster_add",
                "hip_code": test_hip_code,
                "hip_ref_code": test_hip_code,  # Same as ref for testing
                "pytorch_module_code": test_pytorch_code,
                "pytorch_functional_code": test_pytorch_code,
                "atol": 1e-4,
                "rtol": 1e-3
            }
        ]
    }
    
    try:
        print("  Submitting test kernel evaluation...")
        start_time = time.time()
        
        response = requests.post(
            f"{master_url}/run_code_batch",
            json=test_request,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        elapsed = time.time() - start_time
        
        if data.get("responses"):
            result = data["responses"][0]
            compile_ok = result.get("compile_ok", False)
            run_ok = result.get("run_ok", False)
            match_ok = result.get("match_ok", False)
            speedup = result.get("speedup", 0.0)
            
            success = compile_ok  # At minimum, compilation should work
            
            print_result("Kernel evaluation", success)
            print(f"         Compile: {'✓' if compile_ok else '✗'}")
            print(f"         Run: {'✓' if run_ok else '✗'}")
            print(f"         Match: {'✓' if match_ok else '✗'}")
            print(f"         Speedup: {speedup:.2f}x")
            print(f"         Time: {elapsed:.2f}s")
            
            return success
        else:
            print_result("Kernel evaluation", False, "No response received")
            return False
            
    except requests.exceptions.Timeout:
        print_result("Kernel evaluation", False, "Request timed out")
        return False
    except Exception as e:
        print_result("Kernel evaluation", False, str(e))
        return False


def test_batch_evaluation(master_url: str, batch_size: int = 4, verbose: bool = False) -> bool:
    """Test batch kernel evaluation"""
    print_header(f"Testing Batch Evaluation ({batch_size} kernels)")
    
    # Simple HIP kernel template
    hip_template = '''
#include <hip/hip_runtime.h>

extern "C" __global__ void test_scale_kernel_{idx}(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        output[idx] = input[idx] * scale;
    }}
}}
'''
    
    pytorch_template = '''
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = {scale}
    
    def forward(self, x):
        return x * self.scale

def get_inputs():
    return [torch.randn(1024, device='cuda')]

def get_init_inputs():
    return []
'''
    
    requests_list = []
    for i in range(batch_size):
        requests_list.append({
            "kernel_name": f"test_batch_scale_{i}",
            "hip_code": hip_template.format(idx=i),
            "hip_ref_code": hip_template.format(idx=i),
            "pytorch_module_code": pytorch_template.format(scale=2.0 + i),
            "pytorch_functional_code": pytorch_template.format(scale=2.0 + i),
            "atol": 1e-4,
            "rtol": 1e-3
        })
    
    batch_request = {"requests": requests_list}
    
    try:
        print(f"  Submitting {batch_size} kernel evaluations...")
        start_time = time.time()
        
        response = requests.post(
            f"{master_url}/run_code_batch",
            json=batch_request,
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        
        elapsed = time.time() - start_time
        
        responses = data.get("responses", [])
        success_count = sum(1 for r in responses if r.get("compile_ok", False))
        
        all_success = success_count == batch_size
        
        print_result("Batch evaluation", all_success)
        print(f"         Success rate: {success_count}/{batch_size}")
        print(f"         Total time: {elapsed:.2f}s")
        print(f"         Avg per kernel: {elapsed/batch_size:.2f}s")
        
        if verbose:
            print("\n  Individual results:")
            for r in responses:
                name = r.get("kernel_name", "unknown")
                compile_ok = "✓" if r.get("compile_ok") else "✗"
                run_ok = "✓" if r.get("run_ok") else "✗"
                match_ok = "✓" if r.get("match_ok") else "✗"
                print(f"         {name}: compile={compile_ok} run={run_ok} match={match_ok}")
        
        return all_success
        
    except requests.exceptions.Timeout:
        print_result("Batch evaluation", False, "Request timed out")
        return False
    except Exception as e:
        print_result("Batch evaluation", False, str(e))
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test HIP Kernel Evaluation Cluster"
    )
    parser.add_argument(
        "--master",
        type=str,
        default="http://localhost:8080",
        help="Master server URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip kernel evaluation tests (only test connectivity)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for batch evaluation test (default: 4)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  HIP Kernel Evaluation Cluster Test")
    print("=" * 60)
    print(f"  Master URL: {args.master}")
    print("=" * 60)
    
    results = []
    
    # Test 1: Master health
    results.append(("Master Health", test_master_health(args.master, args.verbose)))
    
    if not results[-1][1]:
        print("\n" + "=" * 60)
        print("  ✗ FAILED: Cannot connect to master server")
        print("    Please ensure the master server is running:")
        print("    ./setup_master.sh --config workers.yaml")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    # Test 2: Cluster status
    cluster_data = test_cluster_status(args.master, args.verbose)
    results.append(("Cluster Status", cluster_data.get("ready", False)))
    
    # Test 3: Simple evaluation
    if not args.skip_eval:
        results.append(("Simple Evaluation", test_simple_evaluation(args.master, args.verbose)))
        
        # Test 4: Batch evaluation
        results.append(("Batch Evaluation", test_batch_evaluation(
            args.master, 
            args.batch_size, 
            args.verbose
        )))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        print_result(name, success)
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  ✓ All tests passed! Cluster is ready for use.")
        sys.exit(0)
    else:
        print("\n  ✗ Some tests failed. Please check the configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()

