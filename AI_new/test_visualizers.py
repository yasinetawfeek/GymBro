#!/usr/bin/env python3
"""
Test Script for Model Performance Visualizers

This script tests all the model performance visualizers:
1. Workout Classifier Visualizer
2. Muscle Group Classifier Visualizer  
3. Pose Optimizer Visualizer
4. Combined Model Performance Visualizer

It runs each visualizer with a random video to ensure they work correctly.
"""

import os
import sys
import traceback

def test_workout_classifier_visualizer():
    """Test the workout classifier visualizer"""
    print("=" * 60)
    print("TESTING WORKOUT CLASSIFIER VISUALIZER")
    print("=" * 60)
    
    try:
        from workout_classifier_visualizer import WorkoutClassifierVisualizer
        
        visualizer = WorkoutClassifierVisualizer()
        results = visualizer.run_evaluation()
        
        print("✓ Workout Classifier Visualizer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Workout Classifier Visualizer test failed: {e}")
        traceback.print_exc()
        return False

def test_muscle_group_classifier_visualizer():
    """Test the muscle group classifier visualizer"""
    print("=" * 60)
    print("TESTING MUSCLE GROUP CLASSIFIER VISUALIZER")
    print("=" * 60)
    
    try:
        from muscle_group_classifier_visualizer import MuscleGroupClassifierVisualizer
        
        visualizer = MuscleGroupClassifierVisualizer()
        results = visualizer.run_evaluation()
        
        print("✓ Muscle Group Classifier Visualizer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Muscle Group Classifier Visualizer test failed: {e}")
        traceback.print_exc()
        return False

def test_pose_optimizer_visualizer():
    """Test the pose optimizer visualizer"""
    print("=" * 60)
    print("TESTING POSE OPTIMIZER VISUALIZER")
    print("=" * 60)
    
    try:
        from pose_optimizer_visualizer import PoseOptimizerVisualizer
        
        visualizer = PoseOptimizerVisualizer()
        results = visualizer.run_evaluation()
        
        print("✓ Pose Optimizer Visualizer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Pose Optimizer Visualizer test failed: {e}")
        traceback.print_exc()
        return False

def test_combined_visualizer():
    """Test the combined model performance visualizer"""
    print("=" * 60)
    print("TESTING COMBINED MODEL PERFORMANCE VISUALIZER")
    print("=" * 60)
    
    try:
        from model_performance_visualizer import ModelPerformanceVisualizer
        
        visualizer = ModelPerformanceVisualizer()
        results = visualizer.run_complete_evaluation()
        
        print("✓ Combined Model Performance Visualizer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Combined Model Performance Visualizer test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all visualizer tests"""
    print("STARTING MODEL PERFORMANCE VISUALIZER TESTS")
    print("=" * 80)
    
    test_results = {}
    
    # Test individual visualizers
    test_results['workout_classifier'] = test_workout_classifier_visualizer()
    test_results['muscle_group_classifier'] = test_muscle_group_classifier_visualizer()
    test_results['pose_optimizer'] = test_pose_optimizer_visualizer()
    test_results['combined'] = test_combined_visualizer()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All visualizer tests passed successfully!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return test_results

if __name__ == "__main__":
    main()
