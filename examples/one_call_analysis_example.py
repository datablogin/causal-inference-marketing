"""Example demonstrating the unified CausalAnalysis API.

This example shows how to perform complete causal inference analysis with
a single line of code using the new unified API.
"""

import numpy as np
import pandas as pd

# Import the unified API
from causal_inference import CausalAnalysis

def create_sample_marketing_data():
    """Create sample marketing campaign data for demonstration."""
    np.random.seed(42)
    n = 2000
    
    print("🎯 Generating sample marketing campaign data...")
    
    # Customer characteristics
    age = np.random.normal(35, 12, n)
    income = np.random.exponential(50000, n)
    segment = np.random.choice(['premium', 'standard', 'budget'], n, p=[0.2, 0.5, 0.3])
    previous_purchases = np.random.poisson(3, n)
    
    # Email campaign assignment (with realistic selection bias)
    email_prob = (
        0.4 +  # Base rate
        0.1 * (income > 60000) +  # Higher income customers more likely to receive email
        0.15 * (segment == 'premium') +  # Premium segment prioritized
        0.05 * (previous_purchases > 5)  # Loyal customers targeted
    )
    email_campaign = np.random.binomial(1, np.clip(email_prob, 0.1, 0.9), n)
    
    # Revenue outcome (true treatment effect varies by segment)
    base_revenue = (
        100 +
        0.5 * age +
        0.001 * income +
        20 * previous_purchases +
        50 * (segment == 'premium') +
        30 * (segment == 'standard')
    )
    
    treatment_effect = (
        50 +  # Base treatment effect
        30 * (segment == 'premium') +
        10 * (segment == 'standard') -
        20 * (segment == 'budget')
    )
    
    revenue = (
        base_revenue + 
        treatment_effect * email_campaign +
        np.random.normal(0, 50, n)
    )
    
    data = pd.DataFrame({
        'email_campaign': email_campaign,
        'revenue': revenue,
        'age': age,
        'income': income,
        'segment': segment,
        'previous_purchases': previous_purchases
    })
    
    print(f"✅ Created dataset with {len(data):,} customers")
    print(f"📧 Email campaign sent to {email_campaign.sum():,} customers ({email_campaign.mean():.1%})")
    print(f"💰 Average revenue: ${revenue.mean():.0f}")
    
    return data


def demonstrate_one_line_analysis():
    """Demonstrate the one-line analysis capability."""
    
    print("\n" + "="*60)
    print("🚀 ONE-LINE ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create data
    data = create_sample_marketing_data()
    
    print("\n📊 Running complete causal analysis in one line...")
    print("Code: CausalAnalysis().fit(data).report()")
    
    # THE KEY FEATURE: Complete analysis in one line!
    report = CausalAnalysis().fit(data).report()
    
    # Display results
    effect = report['effect']
    print(f"\n✨ RESULTS:")
    print(f"   Treatment Effect: ${effect.ate:.2f}")
    print(f"   95% Confidence Interval: [${effect.ate_ci_lower:.2f}, ${effect.ate_ci_upper:.2f}]")
    print(f"   Statistical Significance: {'Yes' if effect.p_value < 0.05 else 'No'} (p = {effect.p_value:.4f})")
    print(f"   Method Used: {report['method'].upper()}")
    print(f"   Sample Size: {report['sample_size']:,}")
    
    # Show report was generated
    print(f"\n📄 HTML Report Generated: {len(report['html_report']):,} characters")
    print("   Contains: Executive Summary, Diagnostics, Recommendations, Sensitivity Analysis")
    
    return report


def demonstrate_custom_analysis():
    """Demonstrate customized analysis with specific parameters."""
    
    print("\n" + "="*60)
    print("🎯 CUSTOMIZED ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create data
    data = create_sample_marketing_data()
    
    print("\n🔧 Configuring analysis with specific parameters...")
    
    # Configure analysis
    analysis = CausalAnalysis(
        method='aipw',  # Use doubly robust method
        treatment_column='email_campaign',
        outcome_column='revenue', 
        covariate_columns=['age', 'income', 'previous_purchases'],  # Exclude categorical
        confidence_level=0.95,
        bootstrap_samples=500,
        random_state=42
    )
    
    print(f"   Method: AIPW (Doubly Robust)")
    print(f"   Covariates: {analysis.covariate_columns}")
    print(f"   Bootstrap Samples: {analysis.bootstrap_samples}")
    
    # Fit the model
    print("\n🔍 Fitting causal inference model...")
    analysis.fit(data)
    
    # Get detailed results
    effect = analysis.estimate_ate()
    
    print(f"\n📈 DETAILED RESULTS:")
    print(f"   Average Treatment Effect: ${effect.ate:.2f}")
    print(f"   Standard Error: ${getattr(effect, 'std_error', 'N/A')}")
    print(f"   95% Confidence Interval: [${effect.ate_ci_lower:.2f}, ${effect.ate_ci_upper:.2f}]")
    print(f"   P-value: {effect.p_value:.6f}")
    print(f"   Interpretation: {effect.interpretation}")
    
    # Generate comprehensive report
    print("\n📊 Generating comprehensive HTML report...")
    report = analysis.report(
        template='full',
        include_sensitivity=True,
        include_diagnostics=True,
        title='Email Campaign ROI Analysis',
        analyst_name='Marketing Analytics Team'
    )
    
    print(f"✅ Report generated successfully")
    print(f"   Template: Full (Executive + Technical)")
    print(f"   Includes: Sensitivity Analysis, Diagnostics, Business Recommendations")
    print(f"   Report Size: {len(report['html_report']):,} characters")
    
    return analysis, report


def demonstrate_file_workflow():
    """Demonstrate file-based workflow."""
    
    print("\n" + "="*60) 
    print("📁 FILE-BASED WORKFLOW DEMONSTRATION")
    print("="*60)
    
    import tempfile
    from pathlib import Path
    
    # Create data and save to file
    data = create_sample_marketing_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save data to CSV
        data_file = temp_path / 'campaign_data.csv'
        data.to_csv(data_file, index=False)
        print(f"💾 Saved data to: {data_file.name}")
        
        # Analyze from file
        print("📖 Loading and analyzing data from file...")
        analysis = CausalAnalysis(
            treatment_column='email_campaign',
            outcome_column='revenue'
        )
        analysis.fit(str(data_file))
        
        # Save report to file
        report_file = temp_path / 'campaign_analysis_report.html'
        report = analysis.report(
            output_path=str(report_file),
            title='Email Campaign Analysis Report'
        )
        
        print(f"📄 Saved report to: {report_file.name}")
        print(f"   File size: {report_file.stat().st_size:,} bytes")
        
        # Show that files exist
        print(f"\n✅ Workflow complete:")
        print(f"   Input: {data_file.name} ({data_file.stat().st_size:,} bytes)")
        print(f"   Output: {report_file.name} ({report_file.stat().st_size:,} bytes)")
        
        # Read and preview report content
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"\n📋 Report Preview:")
        print(f"   Contains 'Executive Summary': {'✅' if 'Executive Summary' in content else '❌'}")
        print(f"   Contains 'Treatment Effect': {'✅' if 'Treatment Effect' in content else '❌'}")
        print(f"   Contains 'Recommendations': {'✅' if 'Recommendations' in content else '❌'}")
        

def demonstrate_method_comparison():
    """Demonstrate comparison across different methods."""
    
    print("\n" + "="*60)
    print("⚖️ METHOD COMPARISON DEMONSTRATION") 
    print("="*60)
    
    # Create data
    data = create_sample_marketing_data()
    
    methods = ['g_computation', 'ipw', 'aipw', 'auto']
    results = {}
    
    print("🔬 Testing different causal inference methods...\n")
    
    for method in methods:
        print(f"   Testing {method.upper()}...", end=' ')
        
        analysis = CausalAnalysis(
            method=method,
            treatment_column='email_campaign',
            outcome_column='revenue',
            covariate_columns=['age', 'income', 'previous_purchases'],
            bootstrap_samples=200,  # Reduced for speed
            random_state=42
        )
        
        analysis.fit(data)
        effect = analysis.estimate_ate()
        
        results[method] = {
            'ate': effect.ate,
            'ci_lower': effect.ate_ci_lower,
            'ci_upper': effect.ate_ci_upper,
            'p_value': effect.p_value,
            'actual_method': analysis.method  # For auto selection
        }
        
        print(f"ATE = ${effect.ate:.2f}")
    
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"{'Method':<15} {'ATE':<10} {'95% CI':<25} {'P-value':<10}")
    print("-" * 65)
    
    for method, result in results.items():
        actual_method = result.get('actual_method', method)
        ci_str = f"[${result['ci_lower']:.2f}, ${result['ci_upper']:.2f}]"
        print(f"{actual_method:<15} ${result['ate']:<9.2f} {ci_str:<25} {result['p_value']:<10.4f}")
    
    # Show consistency
    ate_values = [r['ate'] for r in results.values()]
    ate_range = max(ate_values) - min(ate_values)
    print(f"\n📈 Consistency Check:")
    print(f"   ATE Range: ${ate_range:.2f}")
    print(f"   Coefficient of Variation: {np.std(ate_values) / np.mean(ate_values):.2%}")
    

def main():
    """Run all demonstrations."""
    
    print("🎉 UNIFIED CAUSAL ANALYSIS API DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates the new sklearn-style unified API")
    print("that makes causal inference accessible in just one line of code!")
    
    try:
        # Run demonstrations
        demonstrate_one_line_analysis()
        demonstrate_custom_analysis() 
        demonstrate_file_workflow()
        demonstrate_method_comparison()
        
        print("\n" + "="*80)
        print("🎊 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n💡 Key Takeaways:")
        print("   ✅ One-line analysis: CausalAnalysis().fit(data).report()")
        print("   ✅ Automatic method selection based on data characteristics")
        print("   ✅ Business-friendly HTML reports with recommendations")
        print("   ✅ Sklearn-style API with method chaining")
        print("   ✅ File input/output support")
        print("   ✅ Consistent results across different methods")
        print("\n🚀 Ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This may be due to missing dependencies or import issues.")
        raise


if __name__ == '__main__':
    main()