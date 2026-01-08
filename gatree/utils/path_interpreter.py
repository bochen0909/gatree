"""
Path interpretation utilities for GATree decision paths.

This module provides functions to format and display decision paths
returned by the predict_with_path methods.
"""


def format_decision_path(path, feature_names=None):
    """
    Format a decision path into a human-readable string.
    
    Args:
        path (list): Decision path from predict_one_with_path
        feature_names (list, optional): Names of features for better readability
        
    Returns:
        str: Formatted decision path
    """
    if not path:
        return "Empty path"
    
    formatted_steps = []
    
    for i, step in enumerate(path):
        if step['node_type'] == 'internal':
            feature_name = step['feature_name']
            if feature_names and step['feature_index'] < len(feature_names):
                feature_name = feature_names[step['feature_index']]
            
            decision = "YES" if step['decision'] else "NO"
            formatted_step = (
                f"Step {i+1}: {feature_name} = {step['feature_value']:.4f} > {step['threshold']:.4f}? {decision}"
            )
            formatted_steps.append(formatted_step)
            
        elif step['node_type'] == 'leaf':
            formatted_step = f"Final: Predicted value = {step['predicted_value']}"
            formatted_steps.append(formatted_step)
            
        elif step['node_type'] == 'missing_child':
            formatted_step = f"Warning: Missing {step['direction']} child, using default = {step['default_value']}"
            formatted_steps.append(formatted_step)
            
        elif step['node_type'] == 'error':
            formatted_step = f"Error: {step['error_message']}, using default = {step['default_value']}"
            formatted_steps.append(formatted_step)
    
    return "\n".join(formatted_steps)


def print_decision_path(path, feature_names=None, title="Decision Path"):
    """
    Print a formatted decision path.
    
    Args:
        path (list): Decision path from predict_one_with_path
        feature_names (list, optional): Names of features for better readability
        title (str): Title for the output
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(format_decision_path(path, feature_names))
    print()


def analyze_decision_paths(paths, feature_names=None):
    """
    Analyze multiple decision paths to find common patterns.
    
    Args:
        paths (list): List of decision paths
        feature_names (list, optional): Names of features
        
    Returns:
        dict: Analysis results including feature usage frequency, common paths, etc.
    """
    if not paths:
        return {"error": "No paths provided"}
    
    analysis = {
        "total_paths": len(paths),
        "feature_usage": {},
        "path_lengths": [],
        "leaf_values": [],
        "common_decisions": {}
    }
    
    for path in paths:
        path_length = 0
        for step in path:
            if step['node_type'] == 'internal':
                path_length += 1
                feature_idx = step['feature_index']
                feature_name = step['feature_name']
                
                if feature_names and feature_idx < len(feature_names):
                    feature_name = feature_names[feature_idx]
                
                if feature_name not in analysis['feature_usage']:
                    analysis['feature_usage'][feature_name] = 0
                analysis['feature_usage'][feature_name] += 1
                
                # Track common decision patterns
                decision_key = f"{feature_name}_{step['decision']}"
                if decision_key not in analysis['common_decisions']:
                    analysis['common_decisions'][decision_key] = 0
                analysis['common_decisions'][decision_key] += 1
                
            elif step['node_type'] == 'leaf':
                analysis['leaf_values'].append(step['predicted_value'])
        
        analysis['path_lengths'].append(path_length)
    
    # Calculate statistics
    if analysis['path_lengths']:
        analysis['avg_path_length'] = sum(analysis['path_lengths']) / len(analysis['path_lengths'])
        analysis['max_path_length'] = max(analysis['path_lengths'])
        analysis['min_path_length'] = min(analysis['path_lengths'])
    
    # Sort feature usage by frequency
    analysis['feature_usage'] = dict(sorted(
        analysis['feature_usage'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    # Sort common decisions by frequency
    analysis['common_decisions'] = dict(sorted(
        analysis['common_decisions'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    return analysis


def print_path_analysis(analysis):
    """
    Print a formatted analysis of decision paths.
    
    Args:
        analysis (dict): Analysis results from analyze_decision_paths
    """
    print("\nDecision Path Analysis")
    print("=" * 22)
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"Total paths analyzed: {analysis['total_paths']}")
    
    if 'avg_path_length' in analysis:
        print(f"Average path length: {analysis['avg_path_length']:.2f}")
        print(f"Path length range: {analysis['min_path_length']} - {analysis['max_path_length']}")
    
    print("\nFeature Usage Frequency:")
    for feature, count in list(analysis['feature_usage'].items())[:10]:  # Top 10
        percentage = (count / analysis['total_paths']) * 100
        print(f"  {feature}: {count} times ({percentage:.1f}%)")
    
    if len(analysis['feature_usage']) > 10:
        print(f"  ... and {len(analysis['feature_usage']) - 10} more features")
    
    print("\nMost Common Decisions:")
    for decision, count in list(analysis['common_decisions'].items())[:10]:  # Top 10
        percentage = (count / analysis['total_paths']) * 100
        print(f"  {decision}: {count} times ({percentage:.1f}%)")
    
    if analysis['leaf_values']:
        unique_predictions = len(set(analysis['leaf_values']))
        print(f"\nUnique predictions: {unique_predictions}")
        if unique_predictions <= 10:
            from collections import Counter
            pred_counts = Counter(analysis['leaf_values'])
            print("Prediction distribution:")
            for pred, count in pred_counts.most_common():
                percentage = (count / len(analysis['leaf_values'])) * 100
                print(f"  {pred}: {count} times ({percentage:.1f}%)")
    
    print()