"""
Real-Time Pivot Table Matching System
====================================
A comprehensive Python script for real-time data processing, pivot table generation,
and matching operations with reference datasets.

Features:
- Real-time data ingestion from various sources
- Dynamic pivot table generation
- Advanced matching algorithms
- Live visualization and reporting
- Multi-threaded processing for performance
"""

import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import sqlite3
import warnings
from typing import Dict, List, Any, Optional, Tuple
import argparse

warnings.filterwarnings('ignore')

class RealTimePivotMatcher:
    """
    Real-Time Pivot Table Matching System
    Handles dynamic data processing, pivot generation, and matching operations.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the Real-Time Pivot Table Matching System.
        
        Args:
            config_file (str): Path to configuration JSON file
        """
        self.config = self._load_config(config_file)
        self.data_store = {}
        self.reference_data = {}
        self.pivot_cache = {}
        self.match_results = {}
        self.is_running = False
        self.update_interval = self.config.get('update_interval', 5)
        
        # Initialize database
        self._init_database()
        
        # Setup visualization
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def _load_config(self, config_file: str = None) -> Dict:
        """Load configuration from JSON file or use defaults."""
        default_config = {
            'update_interval': 5,
            'data_sources': {
                'csv': ['data/sales_data.csv'],
                'excel': ['data/metrics.xlsx'],
                'api': ['api/sales_api'],
                'database': 'sqlite:data/sales.db'
            },
            'pivot_columns': {
                'rows': ['category', 'region'],
                'columns': ['year', 'quarter'],
                'values': ['sales', 'quantity', 'profit']
            },
            'matching_rules': {
                'similarity_threshold': 0.8,
                'match_columns': ['product_id', 'date', 'customer_id'],
                'fuzzy_match': True,
                'weight_columns': {'sales': 0.5, 'quantity': 0.3, 'profit': 0.2}
            },
            'output': {
                'save_results': True,
                'output_format': ['csv', 'json'],
                'visualization': True
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            return default_config
        return default_config
    
    def _init_database(self):
        """Initialize SQLite database for storing historical data."""
        self.conn = sqlite3.connect('realtime_pivot.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                data_source TEXT,
                pivot_data TEXT,
                match_results TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reference_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                data_type TEXT,
                data_content TEXT
            )
        ''')
        
        self.conn.commit()
    
    def generate_sample_data(self, rows: int = 1000) -> pd.DataFrame:
        """
        Generate sample real-time data for demonstration.
        
        Args:
            rows (int): Number of rows to generate
            
        Returns:
            pd.DataFrame: Generated sample data
        """
        np.random.seed(int(time.time()))
        
        data = {
            'date': pd.date_range(start='2024-01-01', periods=rows, freq='H'),
            'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], rows),
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Sports'], rows),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], rows),
            'sales': np.random.normal(1000, 300, rows).clip(0).round(2),
            'quantity': np.random.poisson(50, rows),
            'profit': np.random.normal(100, 30, rows).clip(0).round(2),
            'customer_id': [f'C{str(i).zfill(4)}' for i in np.random.randint(1, 1001, rows)],
            'channel': np.random.choice(['Online', 'Store', 'Phone', 'Mobile'], rows),
            'discount': np.random.uniform(0, 0.3, rows).round(2)
        }
        
        return pd.DataFrame(data)
    
    def load_data_source(self, source_type: str, source_path: str) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            source_type (str): Type of source ('csv', 'excel', 'database', 'api')
            source_path (str): Path or connection string
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if source_type == 'csv':
                return pd.read_csv(source_path)
            elif source_type == 'excel':
                return pd.read_excel(source_path)
            elif source_type == 'database':
                # Handle database connections
                conn_str = source_path.replace('sqlite:', '')
                conn = sqlite3.connect(conn_str)
                query = f"SELECT * FROM {source_path.split('/')[-1]}"
                return pd.read_sql(query, conn)
            else:
                # Generate sample data for demo
                return self.generate_sample_data()
        except Exception as e:
            print(f"Error loading data from {source_type}: {str(e)}")
            return pd.DataFrame()
    
    def create_pivot_table(self, data: pd.DataFrame, 
                          pivot_config: Dict) -> pd.DataFrame:
        """
        Create pivot table from data based on configuration.
        
        Args:
            data (pd.DataFrame): Input data
            pivot_config (Dict): Pivot configuration
            
        Returns:
            pd.DataFrame: Pivot table
        """
        try:
            # Ensure required columns exist
            required_cols = pivot_config['rows'] + pivot_config['columns'] + pivot_config['values']
            for col in required_cols:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in data")
            
            # Create pivot table
            pivot_table = pd.pivot_table(
                data,
                values=pivot_config['values'],
                index=pivot_config['rows'],
                columns=pivot_config['columns'],
                aggfunc='sum',
                fill_value=0,
                margins=True,
                margins_name='Total'
            )
            
            # Add calculated metrics
            if 'sales' in pivot_table.columns.get_level_values(1):
                pivot_table['sales_profit_ratio'] = (
                    pivot_table[('sales', 'Total')] / pivot_table[('profit', 'Total')].replace(0, np.nan)
                ).fillna(0)
            
            return pivot_table
        
        except Exception as e:
            print(f"Error creating pivot table: {str(e)}")
            return pd.DataFrame()
    
    def calculate_similarity_score(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Calculate similarity score between two dataframes.
        
        Args:
            df1 (pd.DataFrame): First dataframe
            df2 (pd.DataFrame): Second dataframe
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Align dataframes
            common_cols = list(set(df1.columns) & set(df2.columns))
            if not common_cols:
                return 0.0
            
            df1_aligned = df1[common_cols].fillna(0)
            df2_aligned = df2[common_cols].fillna(0)
            
            # Calculate cosine similarity
            dot_product = np.dot(df1_aligned.values.flatten(), df2_aligned.values.flatten())
            norm1 = np.linalg.norm(df1_aligned.values.flatten())
            norm2 = np.linalg.norm(df2_aligned.values.flatten())
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def match_records(self, current_data: pd.DataFrame, 
                     reference_data: pd.DataFrame) -> Dict:
        """
        Match current data against reference data using multiple strategies.
        
        Args:
            current_data (pd.DataFrame): Current dataset
            reference_data (pd.DataFrame): Reference dataset
            
        Returns:
            Dict: Matching results with statistics
        """
        try:
            matching_config = self.config['matching_rules']
            metrics = defaultdict(list)
            matches = []
            
            # Exact matching
            exact_matches = []
            for idx, row in current_data.iterrows():
                match_found = False
                for ref_idx, ref_row in reference_data.iterrows():
                    match_score = 0
                    for col in matching_config['match_columns']:
                        if col in row and col in ref_row:
                            if str(row[col]) == str(ref_row[col]):
                                match_score += matching_config.get('weight_columns', {}).get(col, 1)
                    
                    if match_score > 0:
                        exact_matches.append({
                            'current_index': idx,
                            'reference_index': ref_idx,
                            'match_score': match_score,
                            'match_type': 'exact'
                        })
                        match_found = True
                        break
            
            # Fuzzy matching (if enabled)
            if matching_config.get('fuzzy_match', False):
                from difflib import SequenceMatcher
                
                for idx, row in current_data.iterrows():
                    best_match = None
                    best_score = 0
                    
                    for ref_idx, ref_row in reference_data.iterrows():
                        match_score = 0
                        for col in matching_config['match_columns']:
                            if col in row and col in ref_row:
                                if isinstance(row[col], str) and isinstance(ref_row[col], str):
                                    similarity = SequenceMatcher(None, str(row[col]), str(ref_row[col])).ratio()
                                    match_score += similarity * matching_config.get('weight_columns', {}).get(col, 1)
                        
                        if match_score > best_score:
                            best_score = match_score
                            best_match = ref_idx
                    
                    if best_score > matching_config['similarity_threshold']:
                        matches.append({
                            'current_index': idx,
                            'reference_index': best_match,
                            'match_score': best_score,
                            'match_type': 'fuzzy'
                        })
            
            # Aggregate metrics
            metrics['total_current_records'] = len(current_data)
            metrics['total_reference_records'] = len(reference_data)
            metrics['exact_matches'] = len(exact_matches)
            metrics['fuzzy_matches'] = len([m for m in matches if 'fuzzy' in m['match_type']])
            metrics['total_matches'] = len(exact_matches) + len([m for m in matches if 'fuzzy' in m['match_type']])
            
            if metrics['total_current_records'] > 0:
                metrics['match_rate'] = metrics['total_matches'] / metrics['total_current_records']
            else:
                metrics['match_rate'] = 0.0
            
            return {
                'metrics': dict(metrics),
                'matches': exact_matches + [m for m in matches if 'fuzzy' in m['match_type']],
                'similarity_score': self.calculate_similarity_score(current_data, reference_data)
            }
        
        except Exception as e:
            print(f"Error matching records: {str(e)}")
            return {'metrics': {}, 'matches': [], 'similarity_score': 0.0}
    
    def visualize_pivot_results(self, pivot_table: pd.DataFrame, 
                               title: str = "Pivot Table Visualization"):
        """
        Create visualizations for pivot table results.
        
        Args:
            pivot_table (pd.DataFrame): Pivot table to visualize
            title (str): Title for the visualization
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # 1. Sales by Category and Region
            if ('sales', 'Total') in pivot_table.columns:
                pivot_sales = pivot_table[('sales', 'Total')]
                pivot_sales.unstack('region').plot(kind='bar', ax=axes[0, 0], colormap='viridis')
                axes[0, 0].set_title('Sales by Category and Region')
                axes[0, 0].set_xlabel('Category')
                axes[0, 0].set_ylabel('Sales')
                axes[0, 0].legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 2. Profit Trend
            if ('profit', 'Total') in pivot_table.columns:
                pivot_profit = pivot_table[('profit', 'Total')]
                pivot_profit.unstack('region').T.plot(kind='line', ax=axes[0, 1], marker='o')
                axes[0, 1].set_title('Profit Trend by Region')
                axes[0, 1].set_xlabel('Year-Quarter')
                axes[0, 1].set_ylabel('Profit')
                axes[0, 1].legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 3. Quantity Distribution
            if ('quantity', 'Total') in pivot_table.columns:
                pivot_qty = pivot_table[('quantity', 'Total')]
                pivot_qty.unstack('region').plot(kind='pie', ax=axes[1, 0], autopct='%1.1f%%')
                axes[1, 0].set_title('Quantity Distribution by Region')
                axes[1, 0].set_ylabel('')
            
            # 4. Correlation Heatmap
            numeric_cols = []
            for col in pivot_table.columns:
                if isinstance(col, tuple):
                    if col[0] in ['sales', 'quantity', 'profit']:
                        numeric_cols.append(col[0])
            
            if len(numeric_cols) >= 2:
                correlation_data = pivot_table[numeric_cols].T.corr()
                sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
                axes[1, 1].set_title('Metric Correlations')
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    def process_real_time_data(self):
        """Main function for real-time data processing."""
        print("=" * 60)
        print("REAL-TIME PIVOT TABLE MATCHING SYSTEM")
        print("=" * 60)
        
        # Load reference data
        reference_data = self.load_data_source('csv', self.config['data_sources']['csv'][0])
        if reference_data.empty:
            reference_data = self.generate_sample_data(500)
        
        # Real-time processing loop
        self.is_running = True
        iteration = 0
        
        while self.is_running:
            iteration += 1
            print(f"\n--- PROCESSING ITERATION {iteration} ---")
            
            # Generate or load current data
            current_data = self.load_data_source('csv', self.config['data_sources']['csv'][0])
            if current_data.empty:
                print("Generating sample data...")
                current_data = self.generate_sample_data(500)
            
            print(f"Loaded {len(current_data)} records of current data")
            
            # Create pivot table
            pivot_table = self.create_pivot_table(
                current_data, 
                self.config['pivot_columns']
            )
            
            if not pivot_table.empty:
                print("\nPivot Table Generated:")
                print(pivot_table)
                
                # Visualize results
                if self.config['output']['visualization']:
                    self.visualize_pivot_results(pivot_table, f"Iteration {iteration} - Real-Time Pivot Analysis")
                
                # Match records
                match_results = self.match_records(current_data, reference_data)
                
                print(f"\nMatching Results:")
                print(f"Similarity Score: {match_results['similarity_score']:.3f}")
                print(f"Match Rate: {match_results['metrics'].get('match_rate', 0):.2%}")
                print(f"Total Matches: {match_results['metrics'].get('total_matches', 0)}")
                
                # Store results
                self.pivot_cache[f'iteration_{iteration}'] = pivot_table
                self.match_results[f'iteration_{iteration}'] = match_results
                
                # Save to database
                if self.config['output']['save_results']:
                    self._save_to_database(pivot_table, match_results)
            
            # Wait for next iteration
            time.sleep(self.update_interval)
    
    def _save_to_database(self, pivot_table: pd.DataFrame, match_results: Dict):
        """Save results to database."""
        try:
            cursor = self.conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('''
                INSERT INTO data_logs 
                (timestamp, data_source, pivot_data, match_results)
                VALUES (?, ?, ?, ?)
            ''', (
                timestamp,
                'realtime_processing',
                pivot_table.to_json(),
                json.dumps(match_results)
            ))
            
            self.conn.commit()
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
    
    def generate_report(self, iterations: int = 5):
        """Generate a summary report of recent iterations."""
        try:
            print("\n" + "=" * 60)
            print("REAL-TIME PROCESSING SUMMARY REPORT")
            print("=" * 60)
            
            # Calculate statistics
            if self.match_results:
                match_rates = [r['metrics'].get('match_rate', 0) for r in self.match_results.values()]
                avg_match_rate = np.mean(match_rates) if match_rates else 0
                max_similarity = max([r['similarity_score'] for r in self.match_results.values()])
                min_similarity = min([r['similarity_score'] for r in self.match_results.values()])
                
                print(f"\nProcessing Statistics:")
                print(f"Total Iterations: {len(self.match_results)}")
                print(f"Average Match Rate: {avg_match_rate:.2%}")
                print(f"Max Similarity Score: {max_similarity:.3f}")
                print(f"Min Similarity Score: {min_similarity:.3f}")
                
                # Export results
                if self.config['output']['save_results']:
                    results_df = pd.DataFrame([
                        {
                            'timestamp': k,
                            'match_rate': v['metrics'].get('match_rate', 0),
                            'similarity_score': v['similarity_score']
                        }
                        for k, v in self.match_results.items()
                    ])
                    
                    results_df.to_csv('realtime_pivot_report.csv', index=False)
                    print(f"\nReport saved to: realtime_pivot_report.csv")
            
            # Display recent pivot table
            if self.pivot_cache:
                latest_pivot = list(self.pivot_cache.values())[-1]
                print(f"\nLatest Pivot Table Sample:")
                print(latest_pivot.head())
        
        except Exception as e:
            print(f"Error generating report: {str(e)}")
    
    def stop(self):
        """Stop the real-time processing."""
        self.is_running = False
        if hasattr(self, 'conn'):
            self.conn.close()
        print("\nReal-time processing stopped.")


def main():
    """Main function to run the Real-Time Pivot Table Matching System."""
    parser = argparse.ArgumentParser(description='Real-Time Pivot Table Matching System')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations to run')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with sample data')
    
    args = parser.parse_args()
    
    # Initialize system
    system = RealTimePivotMatcher(args.config)
    
    try:
        if args.demo:
            # Demo mode: Run single iteration with sample data
            print("Running in DEMO MODE with sample data...")
            
            # Generate sample data
            current_data = system.generate_sample_data(1000)
            reference_data = system.generate_sample_data(500)
            
            # Create pivot table
            pivot_table = system.create_pivot_table(current_data, system.config['pivot_columns'])
            
            if not pivot_table.empty:
                print("\nPivot Table Generated:")
                print(pivot_table)
                
                # Visualize
                system.visualize_pivot_results(pivot_table, "Demo - Sample Pivot Analysis")
                
                # Match records
                match_results = system.match_records(current_data, reference_data)
                
                print(f"\nMatching Results:")
                print(f"Similarity Score: {match_results['similarity_score']:.3f}")
                print(f"Total Matches: {match_results['metrics'].get('total_matches', 0)}")
                
                # Generate report
                system.generate_report()
        
        else:
            # Real-time mode
            system.update_interval = args.interval
            system.process_real_time_data()
            
            # Generate final report
            system.generate_report()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping...")
        system.stop()
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        system.stop()


if __name__ == "__main__":
    import os
    main()
