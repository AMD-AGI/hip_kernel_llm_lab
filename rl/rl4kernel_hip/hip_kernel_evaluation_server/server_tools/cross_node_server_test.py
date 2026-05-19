#!/usr/bin/env python3
import requests
import sys

# 替换为实际的 server 节点 IP
SERVER_IP = "10.254.6.40"  # 修改这里
SERVER_PORT = "8080"
SERVER_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

print(f"Testing cross-node connection to {SERVER_URL}")
print("=" * 50)

try:
    # 1. 测试健康检查
    print("1. Testing /health endpoint...")
    response = requests.get(f"{SERVER_URL}/health", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # 2. 测试批量接口（仅测试连接，不测试实际编译）
    print("\n2. Testing /run_code_batch endpoint...")
    test_data = {
        "requests": [
            {
                "kernel_name": "test_connection",
                "hip_code": "// test",
                "hip_ref_code": "// ref",
                "pytorch_module_code": "# test",
                "pytorch_functional_code": "# test"
            }
        ]
    }
    response = requests.post(
        f"{SERVER_URL}/run_code_batch",
        json=test_data,
        timeout=10
    )
    print(f"   HTTP Status: {response.status_code}")
    
    # 解析响应内容
    result = response.json()
    print(f"   Batch Size: {result['batch_size']}")
    print(f"   Total Time: {result['total_time']:.2f}s")
    
    # 检查业务结果（预期编译失败，因为是测试代码）
    if result['responses']:
        resp = result['responses'][0]
        print(f"   Test Result: compile_ok={resp['compile_ok']}, run_ok={resp['run_ok']}, match_ok={resp['match_ok']}")
        print(f"   Note: Compilation expected to fail (using dummy test code)")
    
    print("\n✓ Cross-node connection successful!")
    
except requests.exceptions.ConnectionError:
    print("\n✗ Connection failed! Check:")
    print("  - Server is running")
    print("  - Network connectivity")
    print("  - Firewall settings")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    sys.exit(1)

print("=" * 50)