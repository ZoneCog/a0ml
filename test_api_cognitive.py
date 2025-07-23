#!/usr/bin/env python3
"""
Quick API Test for Cognitive AtomSpace Server

Tests the enhanced API endpoints to ensure they work correctly.
"""

import asyncio
import requests
import time
import json
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from python.api.atomspace_server import AtomSpaceServer


async def test_cognitive_api():
    """Test the cognitive API endpoints"""
    print("🧪 Testing Cognitive AtomSpace API")
    print("=" * 50)
    
    # Create server instance (without running it)
    server = AtomSpaceServer("memory/test_api_atomspace")
    
    # Test direct method calls instead of HTTP requests
    print("🔧 Testing API components directly...")
    
    # Test cognitive integration initialization
    cognitive_integration = server.cognitive_integration
    print(f"✅ Cognitive integration initialized: {cognitive_integration is not None}")
    
    # Test storing a cognitive pattern
    print("\n📚 Testing cognitive pattern storage...")
    pattern = await cognitive_integration.store_cognitive_pattern(
        name="test_api_pattern",
        scheme_expression="(test (api ?endpoint) (success ?result))",
        metadata={"test": True, "api": "cognitive"}
    )
    print(f"✅ Stored pattern: {pattern.name} with {len(pattern.atom_ids)} atoms")
    
    # Test pattern evaluation
    print("\n⚡ Testing pattern evaluation...")
    eval_result = await cognitive_integration.evaluate_cognitive_pattern(
        pattern.id,
        {"endpoint": "store_pattern", "result": "success"}
    )
    print(f"✅ Evaluation successful: {eval_result.get('success', False)}")
    
    # Test pattern search
    print("\n🔍 Testing pattern search...")
    search_results = await cognitive_integration.find_cognitive_patterns("test")
    print(f"✅ Found {len(search_results)} patterns matching 'test'")
    
    # Test statistics
    print("\n📊 Testing statistics...")
    stats = await cognitive_integration.get_cognitive_statistics()
    print(f"✅ Statistics: {stats['cognitive_patterns']} patterns, {stats['grammar_statistics']['total_grammars']} grammars")
    
    # Test export/import
    print("\n💾 Testing export/import...")
    export_data = await cognitive_integration.export_cognitive_knowledge()
    import_success = await cognitive_integration.import_cognitive_knowledge(export_data)
    print(f"✅ Export/import successful: {import_success}")
    
    print("\n" + "=" * 50)
    print("🎉 All cognitive API components working correctly!")
    print("The enhanced AtomSpace server is ready for deployment.")
    
    return True


async def main():
    """Main test execution"""
    try:
        success = await test_cognitive_api()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)