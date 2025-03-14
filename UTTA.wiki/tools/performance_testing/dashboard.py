#!/usr/bin/env python3
"""
Benchmark Dashboard for LlamaIndex Integration

A web-based dashboard for visualizing and analyzing the performance metrics of the
LlamaIndex integration. This dashboard displays benchmark results, quality metrics,
cost analysis, and allows for interactive performance monitoring.
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Try to import streamlit for the dashboard
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("Streamlit not installed. Please install it with 'pip install streamlit plotly'")
    sys.exit(1)

# Import our advanced testing tools
try:
    from advanced_testing_tools import (
        DEFAULT_DB_PATH, DEFAULT_RESULTS_DIR,
        run_comprehensive_benchmark, run_automated_regression_test,
        create_benchmark_database, evaluate_response_quality
    )
    # Import from llama_index_integration for available providers and example queries
    from llama_index_integration import LlamaIndexKnowledgeManager
    
    # Define functions that were previously imported from test_production_llama_index
    def list_available_providers():
        """List available LLM providers based on environment variables."""
        providers = ["local"]  # Local is always available
        if os.environ.get("OPENAI_API_KEY"):
            providers.append("openai")
        if os.environ.get("ANTHROPIC_API_KEY"):
            providers.append("anthropic")
        return providers
    
    def example_queries():
        """Return a list of example educational queries."""
        return [
            "What are effective teaching strategies for ESL students?",
            "How can teachers support students with ADHD?",
            "What are best practices for formative assessment?",
            "How to implement project-based learning?",
            "What strategies help with classroom management?"
        ]
except ImportError as e:
    print(f"Error importing required modules: {str(e)}")
    print("Please make sure all required files are in place and compatible.")
    sys.exit(1)

# Set page configuration
st.set_page_config(
    page_title="LlamaIndex Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_DAYS_BACK = 30

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #306998;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight {
        padding: 1rem;
        background-color: #e6f3ff;
        border-left: 5px solid #4B8BBE;
        margin-bottom: 1rem;
    }
    .warning {
        padding: 1rem;
        background-color: #fff3e6;
        border-left: 5px solid #ff9f43;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_benchmark_data(db_path: Union[str, Path] = DEFAULT_DB_PATH, days_back: int = DEFAULT_DAYS_BACK):
    """Load benchmark data from the SQLite database"""
    try:
        # Calculate date cutoff
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).isoformat()
        
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        
        # Query benchmark runs
        df = pd.read_sql(
            f"SELECT * FROM benchmark_runs WHERE timestamp > '{cutoff_date}'",
            conn
        )
        
        # Query quality metrics
        metrics_df = pd.read_sql(
            f"""
            SELECT bm.id, bm.provider, bm.model, bm.timestamp, qm.metric_name, qm.metric_value
            FROM benchmark_runs bm
            JOIN quality_metrics qm ON bm.id = qm.benchmark_id
            WHERE bm.timestamp > '{cutoff_date}'
            """,
            conn
        )
        
        conn.close()
        
        if len(df) == 0:
            return None, None
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        return df, metrics_df
    
    except Exception as e:
        st.error(f"Error loading benchmark data: {str(e)}")
        return None, None

def display_overview(df):
    """Display overview metrics and charts"""
    st.markdown("<div class='sub-header'>Overview</div>", unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", len(df))
    
    with col2:
        st.metric("Providers", df['provider'].nunique())
    
    with col3:
        st.metric("Avg. Quality", f"{df['overall_quality'].mean():.2f}")
    
    with col4:
        st.metric("Total Cost", f"${df['total_cost'].sum():.4f}")
    
    # Timeline chart
    st.markdown("<div class='sub-header'>Performance Timeline</div>", unsafe_allow_html=True)
    
    timeline_data = df.groupby(['date', 'provider']).agg({
        'execution_time': 'mean',
        'overall_quality': 'mean',
        'total_cost': 'sum'
    }).reset_index()
    
    tab1, tab2, tab3 = st.tabs(["Execution Time", "Quality", "Cost"])
    
    with tab1:
        fig = px.line(
            timeline_data, 
            x="date", 
            y="execution_time", 
            color="provider",
            title="Average Query Execution Time",
            labels={"execution_time": "Execution Time (s)", "date": "Date", "provider": "Provider"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        avg_time_by_provider = df.groupby('provider')['execution_time'].mean().reset_index()
        avg_time_by_provider = avg_time_by_provider.sort_values('execution_time')
        
        fig2 = px.bar(
            avg_time_by_provider,
            x="provider",
            y="execution_time",
            title="Average Execution Time by Provider",
            labels={"execution_time": "Execution Time (s)", "provider": "Provider"},
            color="provider"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        fig = px.line(
            timeline_data, 
            x="date", 
            y="overall_quality", 
            color="provider",
            title="Average Response Quality",
            labels={"overall_quality": "Quality Score", "date": "Date", "provider": "Provider"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        avg_quality_by_provider = df.groupby('provider')['overall_quality'].mean().reset_index()
        avg_quality_by_provider = avg_quality_by_provider.sort_values('overall_quality', ascending=False)
        
        fig2 = px.bar(
            avg_quality_by_provider,
            x="provider",
            y="overall_quality",
            title="Average Quality by Provider",
            labels={"overall_quality": "Quality Score", "provider": "Provider"},
            color="provider"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        fig = px.line(
            timeline_data, 
            x="date", 
            y="total_cost", 
            color="provider",
            title="Daily API Cost",
            labels={"total_cost": "Cost (USD)", "date": "Date", "provider": "Provider"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        total_cost_by_provider = df.groupby('provider')['total_cost'].sum().reset_index()
        total_cost_by_provider = total_cost_by_provider.sort_values('total_cost', ascending=False)
        
        fig2 = px.bar(
            total_cost_by_provider,
            x="provider",
            y="total_cost",
            title="Total Cost by Provider",
            labels={"total_cost": "Cost (USD)", "provider": "Provider"},
            color="provider"
        )
        st.plotly_chart(fig2, use_container_width=True)

def display_provider_comparison(df, metrics_df):
    """Display provider comparison metrics and charts"""
    st.markdown("<div class='sub-header'>Provider Comparison</div>", unsafe_allow_html=True)
    
    # Get available providers
    providers = df['provider'].unique().tolist()
    
    if len(providers) < 2:
        st.info("Need at least two providers for comparison. Please run benchmarks with multiple providers.")
        return
    
    # Provider selection
    selected_providers = st.multiselect(
        "Select providers to compare:",
        options=providers,
        default=providers[:2]
    )
    
    if len(selected_providers) < 2:
        st.warning("Please select at least two providers for comparison.")
        return
    
    # Filter data for selected providers
    filtered_df = df[df['provider'].isin(selected_providers)]
    
    # Performance metrics
    performance_metrics = filtered_df.groupby('provider').agg({
        'execution_time': ['mean', 'min', 'max'],
        'overall_quality': ['mean', 'min', 'max'],
        'total_cost': ['mean', 'sum'],
        'id': 'count'
    }).reset_index()
    
    # Fix column names
    performance_metrics.columns = [
        'provider', 'avg_time', 'min_time', 'max_time', 
        'avg_quality', 'min_quality', 'max_quality',
        'avg_cost', 'total_cost', 'query_count'
    ]
    
    # Display metrics table
    st.markdown("### Performance Metrics")
    st.dataframe(
        performance_metrics[['provider', 'query_count', 'avg_time', 'avg_quality', 'avg_cost', 'total_cost']]
        .set_index('provider')
        .style.format({
            'avg_time': '{:.2f}',
            'avg_quality': '{:.2f}',
            'avg_cost': '${:.5f}',
            'total_cost': '${:.4f}'
        })
    )
    
    # Quality metrics comparison
    st.markdown("### Quality Metrics Comparison")
    
    if metrics_df is not None:
        # Filter metrics for selected providers
        filtered_metrics = metrics_df[metrics_df['provider'].isin(selected_providers)]
        
        # Pivot table for quality metrics
        metrics_pivot = filtered_metrics.pivot_table(
            index='provider',
            columns='metric_name',
            values='metric_value',
            aggfunc='mean'
        ).reset_index()
        
        # Radar chart of quality metrics
        categories = [col for col in metrics_pivot.columns if col != 'provider']
        
        fig = go.Figure()
        
        for i, provider in enumerate(metrics_pivot['provider']):
            values = metrics_pivot.iloc[i][categories].tolist()
            values.append(values[0])  # Close the loop
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],  # Close the loop
                fill='toself',
                name=provider
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Quality Metrics by Provider"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart comparison for individual metrics
        selected_metric = st.selectbox(
            "Select metric for detailed comparison:",
            options=categories
        )
        
        metric_data = filtered_metrics[filtered_metrics['metric_name'] == selected_metric]
        avg_metric_by_provider = metric_data.groupby('provider')['metric_value'].mean().reset_index()
        
        fig = px.bar(
            avg_metric_by_provider,
            x="provider",
            y="metric_value",
            title=f"{selected_metric} by Provider",
            labels={"metric_value": "Score", "provider": "Provider"},
            color="provider"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_cost_analysis(df):
    """Display cost analysis and token usage"""
    st.markdown("<div class='sub-header'>Cost Analysis</div>", unsafe_allow_html=True)
    
    # Total cost over time
    daily_cost = df.groupby(['date', 'provider'])['total_cost'].sum().reset_index()
    
    fig = px.area(
        daily_cost, 
        x="date", 
        y="total_cost", 
        color="provider",
        title="Daily API Cost by Provider",
        labels={"total_cost": "Cost (USD)", "date": "Date", "provider": "Provider"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost breakdown by model
    model_cost = df.groupby(['provider', 'model'])['total_cost'].sum().reset_index()
    
    fig = px.sunburst(
        model_cost,
        path=['provider', 'model'],
        values='total_cost',
        title="Cost Distribution by Provider and Model",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Token usage analysis
    st.markdown("### Token Usage Analysis")
    
    token_data = df.groupby('provider').agg({
        'query_tokens': 'sum',
        'response_tokens': 'sum',
        'total_cost': 'sum'
    }).reset_index()
    
    token_data['total_tokens'] = token_data['query_tokens'] + token_data['response_tokens']
    token_data['cost_per_1k_tokens'] = 1000 * token_data['total_cost'] / token_data['total_tokens']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            token_data,
            x="provider",
            y=["query_tokens", "response_tokens"],
            title="Token Usage by Provider",
            labels={"value": "Token Count", "provider": "Provider", "variable": "Token Type"},
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            token_data,
            x="provider",
            y="cost_per_1k_tokens",
            title="Effective Cost per 1K Tokens",
            labels={"cost_per_1k_tokens": "Cost (USD)", "provider": "Provider"},
            color="provider"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_insights(df):
    """Display insights and recommendations based on benchmark data"""
    st.markdown("<div class='sub-header'>Insights & Recommendations</div>", unsafe_allow_html=True)
    
    # Identify best provider for quality
    quality_data = df.groupby('provider')['overall_quality'].mean().reset_index()
    best_quality_provider = quality_data.loc[quality_data['overall_quality'].idxmax()]['provider']
    best_quality_score = quality_data.loc[quality_data['overall_quality'].idxmax()]['overall_quality']
    
    # Identify fastest provider
    speed_data = df.groupby('provider')['execution_time'].mean().reset_index()
    fastest_provider = speed_data.loc[speed_data['execution_time'].idxmin()]['provider']
    fastest_speed = speed_data.loc[speed_data['execution_time'].idxmin()]['execution_time']
    
    # Identify most cost-effective provider
    # Calculate cost per quality point
    cost_effectiveness = df.groupby('provider').agg({
        'total_cost': 'sum',
        'overall_quality': 'mean'
    }).reset_index()
    
    cost_effectiveness['cost_per_quality'] = cost_effectiveness['total_cost'] / cost_effectiveness['overall_quality']
    best_value_provider = cost_effectiveness.loc[cost_effectiveness['cost_per_quality'].idxmin()]['provider']
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight">
            <h3>Quality Champion: {best_quality_provider}</h3>
            <p>Based on benchmark data, <strong>{best_quality_provider}</strong> provides the highest quality 
            responses with an average score of {best_quality_score:.2f}.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight">
            <h3>Speed Leader: {fastest_provider}</h3>
            <p><strong>{fastest_provider}</strong> has the fastest average response time at {fastest_speed:.2f} seconds.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight">
            <h3>Best Value: {best_value_provider}</h3>
            <p>For the best balance of quality and cost, <strong>{best_value_provider}</strong> provides 
            the most cost-effective option based on quality per dollar spent.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check for any concerning trends
        recent_df = df[df['timestamp'] > (datetime.datetime.now() - datetime.timedelta(days=7))]
        if len(recent_df) > 0:
            recent_quality = recent_df['overall_quality'].mean()
            overall_quality = df['overall_quality'].mean()
            
            if recent_quality < overall_quality * 0.9:  # 10% drop in quality
                st.markdown(f"""
                <div class="warning">
                    <h3>Quality Alert</h3>
                    <p>There has been a {((overall_quality - recent_quality) / overall_quality * 100):.1f}% drop in 
                    overall response quality in the past 7 days. Consider investigating potential issues.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### Recommendations")
    
    # Create recommendations based on data
    recommendations = []
    
    # Quality recommendation
    if quality_data['overall_quality'].max() < 0.7:
        recommendations.append(
            "Consider tuning retrieval parameters to improve overall response quality. " + 
            "Increasing context relevance with more precise chunk sizes may help."
        )
    
    # Speed recommendation
    if speed_data['execution_time'].min() > 5.0:
        recommendations.append(
            "Response times are higher than optimal. Consider implementing more aggressive caching " +
            "or using a faster model for initial responses."
        )
    
    # Cost recommendation
    if cost_effectiveness['cost_per_quality'].min() > 0.1:
        recommendations.append(
            "Cost efficiency could be improved. Consider testing with different models " +
            "or adjusting token usage to optimize cost-effectiveness."
        )
    
    # Display recommendations
    for i, recommendation in enumerate(recommendations):
        st.markdown(f"**{i+1}. {recommendation}**")
    
    if not recommendations:
        st.success("Your LlamaIndex integration is performing well! Continue monitoring for any changes.")

def run_new_benchmark():
    """Interface for running a new benchmark"""
    st.markdown("<div class='sub-header'>Run New Benchmark</div>", unsafe_allow_html=True)
    
    with st.form("benchmark_form"):
        # Provider selection
        available_providers = list_available_providers()
        selected_providers = st.multiselect(
            "Select providers to benchmark:",
            options=available_providers,
            default=available_providers if available_providers else []
        )
        
        # Query selection
        default_queries = example_queries()
        
        query_type = st.radio(
            "Query Source:",
            options=["Example Queries", "Custom Queries"]
        )
        
        if query_type == "Example Queries":
            selected_queries = st.multiselect(
                "Select example queries to use:",
                options=default_queries,
                default=default_queries[:2] if default_queries else []
            )
        else:
            custom_queries = st.text_area(
                "Enter custom queries (one per line):",
                height=150
            )
            selected_queries = [q.strip() for q in custom_queries.split('\n') if q.strip()]
        
        # Additional settings
        col1, col2 = st.columns(2)
        
        with col1:
            documents_dir = st.text_input(
                "Documents Directory:",
                value=os.environ.get("KNOWLEDGE_DIR", "")
            )
            
            top_k = st.slider(
                "Top K Results:",
                min_value=1,
                max_value=20,
                value=5
            )
        
        with col2:
            verbose = st.checkbox("Verbose Output", value=True)
            save_results = st.checkbox("Save Results to Database", value=True)
        
        submit = st.form_submit_button("Run Benchmark")
    
    if submit:
        if not selected_providers:
            st.error("Please select at least one provider.")
            return
        
        if not selected_queries:
            st.error("Please select or enter at least one query.")
            return
        
        # Display progress
        progress = st.progress(0)
        status = st.empty()
        
        # Run benchmark
        try:
            status.text("Initializing benchmark...")
            
            results = run_comprehensive_benchmark(
                queries=selected_queries,
                providers=selected_providers,
                documents_dir=documents_dir if documents_dir else None,
                top_k=top_k,
                verbose=verbose,
                save_results=save_results
            )
            
            # Display results summary
            st.success("Benchmark completed successfully!")
            
            # Create a summary table
            summary_data = []
            
            for provider in results:
                provider_times = []
                provider_qualities = []
                
                for query in results[provider]:
                    result = results[provider][query]
                    provider_times.append(result.get("execution_time", 0))
                    
                    if "response" in result and "sources" in result:
                        quality = evaluate_response_quality(
                            query,
                            result["response"],
                            result.get("sources", [])
                        )
                        provider_qualities.append(quality.get("overall_score", 0))
                
                avg_time = sum(provider_times) / len(provider_times) if provider_times else 0
                avg_quality = sum(provider_qualities) / len(provider_qualities) if provider_qualities else 0
                
                summary_data.append({
                    "provider": provider,
                    "queries": len(results[provider]),
                    "avg_time": avg_time,
                    "avg_quality": avg_quality
                })
            
            # Display summary table
            st.dataframe(
                pd.DataFrame(summary_data)
                .style.format({
                    'avg_time': '{:.2f}',
                    'avg_quality': '{:.2f}'
                })
            )
            
            # Recommend refresh
            st.info("Refresh the dashboard to see updated benchmark results in the main view.")
            
        except Exception as e:
            st.error(f"Error running benchmark: {str(e)}")
        finally:
            progress.empty()

def regression_test_interface():
    """Interface for running regression tests"""
    st.markdown("<div class='sub-header'>Regression Testing</div>", unsafe_allow_html=True)
    
    with st.form("regression_form"):
        # Provider selection
        available_providers = list_available_providers()
        selected_provider = st.selectbox(
            "Select provider to test:",
            options=available_providers
        )
        
        # Test settings
        col1, col2 = st.columns(2)
        
        with col1:
            documents_dir = st.text_input(
                "Documents Directory:",
                value=os.environ.get("KNOWLEDGE_DIR", "")
            )
            
            sample_size = st.slider(
                "Sample Size:",
                min_value=1,
                max_value=10,
                value=3
            )
        
        with col2:
            threshold = st.slider(
                "Quality Threshold:",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05
            )
            
            verbose = st.checkbox("Verbose Output", value=True)
        
        submit = st.form_submit_button("Run Regression Test")
    
    if submit:
        if not selected_provider:
            st.error("Please select a provider.")
            return
        
        # Display progress
        status = st.empty()
        
        # Run regression test
        try:
            status.text("Running regression test...")
            
            result = run_automated_regression_test(
                documents_dir=documents_dir if documents_dir else None,
                provider=selected_provider,
                sample_size=sample_size,
                threshold=threshold,
                verbose=verbose
            )
            
            # Display results
            if "error" in result:
                st.error(result["error"])
                return
            
            # Create result visualization
            if result["regression_detected"]:
                st.error(f"⚠️ Regression detected! {result['queries_passed']} of {result['queries_tested']} queries passed.")
            else:
                st.success(f"✅ No regression detected. All {result['queries_passed']} of {result['queries_tested']} queries passed.")
            
            # Display detailed results
            st.markdown("### Test Details")
            for query, details in result["details"].items():
                with st.expander(f"Query: {query}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Baseline Quality", f"{details['baseline_quality']:.2f}")
                    
                    with col2:
                        st.metric("Current Quality", f"{details['current_quality']:.2f}")
                    
                    with col3:
                        delta = f"{(details['quality_ratio'] - 1) * 100:.1f}%" if details['quality_ratio'] > 1 else f"{(1 - details['quality_ratio']) * 100:.1f}%"
                        st.metric(
                            "Quality Ratio", 
                            f"{details['quality_ratio']:.2f}",
                            delta=delta if details['quality_ratio'] >= 1 else f"-{delta}"
                        )
                    
                    st.markdown(f"**Result:** {'✅ Passed' if details['passed'] else '❌ Failed'}")
                    
                    # Time comparison
                    st.markdown("#### Time Comparison")
                    time_data = pd.DataFrame([
                        {"version": "Baseline", "time": details["baseline_time"]},
                        {"version": "Current", "time": details["current_time"]}
                    ])
                    
                    fig = px.bar(
                        time_data,
                        x="version",
                        y="time",
                        title="Execution Time Comparison",
                        labels={"time": "Time (s)", "version": "Version"}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error running regression test: {str(e)}")
        finally:
            status.empty()

def main():
    """Main dashboard function"""
    # Header
    st.markdown("<div class='main-header'>LlamaIndex Performance Dashboard</div>", unsafe_allow_html=True)
    st.markdown("Monitor and analyze the performance of your LlamaIndex integration")
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.llamaindex.ai/hubfs/logo-1.svg", width=200)
        st.markdown("## Dashboard Controls")
        
        # Database selection
        db_path = st.text_input(
            "Benchmark Database Path:",
            value=str(DEFAULT_DB_PATH)
        )
        
        # Date range selection
        days_back = st.slider(
            "Days to Analyze:",
            min_value=1,
            max_value=90,
            value=DEFAULT_DAYS_BACK
        )
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation:",
            options=[
                "📊 Overview", 
                "🔍 Provider Comparison", 
                "💰 Cost Analysis", 
                "💡 Insights", 
                "🧪 Run New Benchmark",
                "🔄 Regression Testing"
            ]
        )
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.experimental_rerun()
    
    # Load data
    df, metrics_df = load_benchmark_data(db_path, days_back)
    
    if df is None and page not in ["🧪 Run New Benchmark", "🔄 Regression Testing"]:
        # Check if database exists
        if not Path(db_path).exists():
            st.warning(f"Benchmark database not found at: {db_path}")
            st.info("Run a benchmark first to generate performance data.")
            
            # Show benchmark page
            run_new_benchmark()
        else:
            st.warning(f"No benchmark data found for the last {days_back} days.")
            
            # Offer to run new benchmark
            if st.button("Run a New Benchmark"):
                page = "🧪 Run New Benchmark"
    else:
        # Display selected page
        if page == "📊 Overview":
            display_overview(df)
        
        elif page == "🔍 Provider Comparison":
            display_provider_comparison(df, metrics_df)
        
        elif page == "💰 Cost Analysis":
            display_cost_analysis(df)
        
        elif page == "💡 Insights":
            display_insights(df)
        
        elif page == "🧪 Run New Benchmark":
            run_new_benchmark()
        
        elif page == "🔄 Regression Testing":
            regression_test_interface()

if __name__ == "__main__":
    main() 