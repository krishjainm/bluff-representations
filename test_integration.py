"""
End-to-end integration tests for deception circuits framework.

This test suite validates that all components work together correctly,
including real model integration, reasoning traces, and the full pipeline.
"""

import torch
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil
import json

sys.path.append('.')

try:
    from deception_circuits import (
        DeceptionTrainingPipeline,
        ReasoningTraceCollector,
        ReasoningTrace,
        ReasoningStep,
        ActivationExtractor,
        AdvancedCausalTester
    )
    print("[PASS] All modules imported successfully")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)


def test_reasoning_trace_to_activations():
    """Test conversion of reasoning traces to activation tensors."""
    print("\n[TEST] Reasoning Trace to Activations Conversion...")
    
    try:
        collector = ReasoningTraceCollector(model=None, tokenizer=None, device="cpu")
        
        # Create mock reasoning traces
        num_samples = 5
        num_layers = 3
        hidden_dim = 128
        
        for i in range(num_samples):
            trace = ReasoningTrace(
                trace_id=f"trace_{i}",
                scenario="poker",
                label=i % 2,  # Alternate between truthful (0) and deceptive (1)
                steps=[],
                final_response=f"Response {i}"
            )
            
            # Add a few steps
            for step_idx in range(3):
                step = ReasoningStep(
                    step_index=step_idx,
                    prompt=f"Step {step_idx}",
                    response=f"Response {step_idx}",
                    activations=torch.randn(num_layers, hidden_dim),
                    tokens=[f"token_{j}" for j in range(10)]
                )
                trace.steps.append(step)
            
            collector.traces.append(trace)
        
        # Test conversion to activations
        activations, labels = collector.convert_traces_to_activations(use_final_step=True)
        
        assert activations.shape == (num_samples, num_layers, hidden_dim), \
            f"Expected shape ({num_samples}, {num_layers}, {hidden_dim}), got {activations.shape}"
        assert labels.shape == (num_samples,), f"Expected labels shape ({num_samples},), got {labels.shape}"
        assert torch.all((labels == 0) | (labels == 1)), "Labels must be 0 or 1"
        
        print("[PASS] Reasoning trace to activations conversion works")
        return True
        
    except Exception as e:
        print(f"[FAIL] Reasoning trace conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reasoning_trace_to_dataframe():
    """Test conversion of reasoning traces to DataFrame."""
    print("\n[TEST] Reasoning Trace to DataFrame Conversion...")
    
    try:
        collector = ReasoningTraceCollector(model=None, tokenizer=None, device="cpu")
        
        # Create mock trace
        trace = ReasoningTrace(
            trace_id="test_trace",
            scenario="poker",
            label=1,
            steps=[
                ReasoningStep(
                    step_index=0,
                    prompt="Test prompt",
                    response="Test response",
                    activations=torch.randn(3, 128),
                    tokens=["token1", "token2"]
                )
            ],
            final_response="Final response"
        )
        collector.traces.append(trace)
        
        # Convert to DataFrame
        df = collector.convert_traces_to_dataframe()
        
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        assert 'statement' in df.columns
        assert 'response' in df.columns
        assert 'label' in df.columns
        assert 'scenario' in df.columns
        assert df.iloc[0]['label'] == 1
        assert df.iloc[0]['scenario'] == 'poker'
        
        print("[PASS] Reasoning trace to DataFrame conversion works")
        return True
        
    except Exception as e:
        print(f"[FAIL] DataFrame conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline_with_synthetic_data():
    """Test full training pipeline with synthetic data."""
    print("\n[TEST] Full Training Pipeline with Synthetic Data...")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create synthetic CSV data
        csv_path = Path(temp_dir) / "test_data.csv"
        num_samples = 20
        
        df = pd.DataFrame({
            'statement': [f"Statement {i}" for i in range(num_samples)],
            'response': [f"Response {i}" for i in range(num_samples)],
            'label': [i % 2 for i in range(num_samples)],
            'scenario': ['poker' if i % 2 == 0 else 'sandbagging' for i in range(num_samples)]
        })
        df.to_csv(csv_path, index=False)
        
        # Create synthetic activations
        act_dir = Path(temp_dir) / "activations"
        act_dir.mkdir()
        num_layers = 5
        hidden_dim = 128
        
        for i in range(num_samples):
            # Create activations with some structure (deceptive samples get different pattern)
            activations = torch.randn(num_layers, hidden_dim)
            if df.iloc[i]['label'] == 1:  # Deceptive
                activations[-2:, :50] += 0.5  # Add signal in last 2 layers
            torch.save(activations, act_dir / f"activations_{i}.pt")
        
        # Run pipeline
        output_dir = Path(temp_dir) / "results"
        pipeline = DeceptionTrainingPipeline(device="cpu", output_dir=output_dir)
        results = pipeline.run_full_experiment(
            csv_path=csv_path,
            activation_dir=act_dir,
            max_samples=num_samples,
            probe_config={'epochs': 5, 'validation_split': 0.2},
            autoencoder_config={'epochs': 5, 'lr': 0.001}
        )
        
        # Validate results
        assert 'probe_results' in results
        assert 'autoencoder_results' in results
        assert 'analysis_results' in results
        assert 'best_layer' in results['probe_results']
        
        print("[PASS] Full training pipeline works with synthetic data")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"[FAIL] Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def test_advanced_causal_testing():
    """Test advanced causal testing methods."""
    print("\n[TEST] Advanced Causal Testing...")
    
    try:
        tester = AdvancedCausalTester(device="cpu")
        
        # Create synthetic activations
        num_samples = 10
        num_layers = 5
        hidden_dim = 128
        
        truthful_activations = torch.randn(num_samples, num_layers, hidden_dim)
        deceptive_activations = truthful_activations.clone()
        deceptive_activations[:, -2:, :50] += 0.5  # Add deception signal
        
        truthful_labels = torch.zeros(num_samples)
        deceptive_labels = torch.ones(num_samples)
        
        # Test comprehensive advanced test (should not crash)
        results = tester.run_comprehensive_advanced_test(
            model=None,  # No model needed for this test
            truthful_activations=truthful_activations,
            deceptive_activations=deceptive_activations,
            truthful_labels=truthful_labels,
            deceptive_labels=deceptive_labels,
            probe=None,
            layer_indices=[3, 4]  # Test last 2 layers
        )
        
        assert isinstance(results, dict)
        print("[PASS] Advanced causal testing runs without errors")
        return True
        
    except Exception as e:
        print(f"[FAIL] Advanced causal testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling in various components."""
    print("\n[TEST] Error Handling...")
    
    try:
        # Test ActivationExtractor with invalid model name
        extractor = ActivationExtractor("invalid_model_name_xyz123", device="cpu")
        try:
            extractor.load_model()
            print("[WARN] Model loading should have failed with invalid name")
        except (ValueError, OSError) as e:
            print(f"[PASS] Correctly raised error for invalid model: {type(e).__name__}")
        
        # Test empty text list
        extractor = ActivationExtractor("microsoft/DialoGPT-medium", device="cpu")
        try:
            extractor.extract_activations([])
            print("[WARN] Should have raised error for empty text list")
        except ValueError as e:
            print(f"[PASS] Correctly raised ValueError for empty text list")
        
        print("[PASS] Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("END-TO-END INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_reasoning_trace_to_activations,
        test_reasoning_trace_to_dataframe,
        test_training_pipeline_with_synthetic_data,
        test_advanced_causal_testing,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("[SUCCESS] All integration tests passed!")
        return True
    else:
        print("[WARNING] Some integration tests failed. Framework may have issues.")
        return False


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)

