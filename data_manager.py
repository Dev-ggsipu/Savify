import pandas as pd
import csv
import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any

class DataManager:
    def __init__(self):
        self.expenses_file = "expenses.csv"
        self.income_file = "income.csv"
        self.budget_file = "budgets.csv"
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize CSV files if they don't exist"""
        # Initialize expenses file
        if not os.path.exists(self.expenses_file):
            with open(self.expenses_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['date', 'description', 'amount', 'category'])
        
        # Initialize income file
        if not os.path.exists(self.income_file):
            with open(self.income_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['year', 'month', 'income'])
        
        # Initialize budget file
        if not os.path.exists(self.budget_file):
            with open(self.budget_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['year', 'month', 'category', 'budget_amount'])
    
    def add_expense(self, description: str, amount: float, category: str, expense_date: date) -> bool:
        """Add a new expense to the CSV file"""
        try:
            with open(self.expenses_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    expense_date.strftime('%Y-%m-%d'),
                    description,
                    f"{amount:.2f}",
                    category
                ])
            return True
        except Exception as e:
            print(f"Error adding expense: {e}")
            return False
    
    def get_all_expenses(self) -> pd.DataFrame:
        """Get all expenses from the CSV file"""
        try:
            if os.path.getsize(self.expenses_file) > 0:
                df = pd.read_csv(self.expenses_file)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df['amount'] = pd.to_numeric(df['amount'])
                    return df
            return pd.DataFrame(columns=['date', 'description', 'amount', 'category'])
        except Exception as e:
            print(f"Error reading expenses: {e}")
            return pd.DataFrame(columns=['date', 'description', 'amount', 'category'])
    
    def get_expenses_by_month(self, year: int, month: int) -> pd.DataFrame:
        """Get expenses for a specific month and year"""
        all_expenses = self.get_all_expenses()
        
        if all_expenses.empty:
            return pd.DataFrame(columns=['date', 'description', 'amount', 'category'])
        
        # Filter by year and month
        filtered_expenses = all_expenses[
            (all_expenses['date'].dt.year == year) & 
            (all_expenses['date'].dt.month == month)
        ]
        
        return filtered_expenses.reset_index(drop=True)
    
    def set_monthly_income(self, year: int, month: int, income: float) -> bool:
        """Set or update monthly income"""
        try:
            # Read existing income data
            income_df = self.get_income_history()
            
            # Check if record exists for this year/month
            existing_record = income_df[
                (income_df['year'] == year) & (income_df['month'] == month)
            ]
            
            if not existing_record.empty:
                # Update existing record
                income_df.loc[
                    (income_df['year'] == year) & (income_df['month'] == month), 
                    'income'
                ] = income
            else:
                # Add new record
                new_record = pd.DataFrame({
                    'year': [year],
                    'month': [month], 
                    'income': [income]
                })
                if income_df.empty:
                    income_df = new_record
                else:
                    income_df = pd.concat([income_df, new_record], ignore_index=True)
            
            # Save back to CSV
            income_df.to_csv(self.income_file, index=False)
            return True
            
        except Exception as e:
            print(f"Error setting monthly income: {e}")
            return False
    
    def get_monthly_income(self, year: int, month: int) -> float:
        """Get monthly income for a specific year and month"""
        try:
            income_df = self.get_income_history()
            
            if income_df.empty:
                return 0.0
            
            # Filter for specific year and month
            monthly_income = income_df[
                (income_df['year'] == year) & (income_df['month'] == month)
            ]
            
            if not monthly_income.empty:
                return float(monthly_income.iloc[0]['income'])
            
            return 0.0
            
        except Exception as e:
            print(f"Error getting monthly income: {e}")
            return 0.0
    
    def get_income_history(self) -> pd.DataFrame:
        """Get all income records"""
        try:
            if os.path.exists(self.income_file) and os.path.getsize(self.income_file) > 0:
                df = pd.read_csv(self.income_file)
                if not df.empty:
                    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                    df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
                    df['income'] = pd.to_numeric(df['income'], errors='coerce')
                    
                    # Remove any rows with invalid data
                    df = df.dropna()
                    
                    # Sort by year and month (most recent first)
                    df = df.sort_values(['year', 'month'], ascending=[False, False])
                    
                    return df
            
            return pd.DataFrame(columns=['year', 'month', 'income'])
            
        except Exception as e:
            print(f"Error reading income history: {e}")
            return pd.DataFrame(columns=['year', 'month', 'income'])
    
    def get_expense_categories(self) -> List[str]:
        """Get list of unique expense categories"""
        try:
            expenses_df = self.get_all_expenses()
            if not expenses_df.empty:
                return sorted(expenses_df['category'].unique().tolist())
            return []
        except Exception as e:
            print(f"Error getting expense categories: {e}")
            return []
    
    def get_monthly_summary(self, year: int, month: int) -> Dict[str, Any]:
        """Get comprehensive monthly financial summary"""
        try:
            expenses_df = self.get_expenses_by_month(year, month)
            monthly_income = self.get_monthly_income(year, month)
            
            total_expenses = expenses_df['amount'].sum() if not expenses_df.empty else 0
            remaining_budget = monthly_income - total_expenses
            
            # Category breakdown
            category_breakdown = {}
            if not expenses_df.empty:
                category_breakdown = expenses_df.groupby('category')['amount'].sum().to_dict()
            
            return {
                'income': monthly_income,
                'total_expenses': total_expenses,
                'remaining_budget': remaining_budget,
                'expense_count': len(expenses_df),
                'category_breakdown': category_breakdown,
                'is_over_budget': remaining_budget < 0
            }
            
        except Exception as e:
            print(f"Error getting monthly summary: {e}")
            return {
                'income': 0.0,
                'total_expenses': 0.0,
                'remaining_budget': 0.0,
                'expense_count': 0,
                'category_breakdown': {},
                'is_over_budget': False
            }
    
    def set_category_budget(self, year: int, month: int, category: str, budget_amount: float) -> bool:
        """Set or update budget for a specific category in a month"""
        try:
            # Read existing budget data
            budget_df = self.get_all_budgets()
            
            # Check if record exists for this year/month/category
            existing_record = budget_df[
                (budget_df['year'] == year) & 
                (budget_df['month'] == month) & 
                (budget_df['category'] == category)
            ]
            
            if not existing_record.empty:
                # Update existing record
                budget_df.loc[
                    (budget_df['year'] == year) & 
                    (budget_df['month'] == month) & 
                    (budget_df['category'] == category), 
                    'budget_amount'
                ] = budget_amount
            else:
                # Add new record
                new_record = pd.DataFrame({
                    'year': [year],
                    'month': [month],
                    'category': [category],
                    'budget_amount': [budget_amount]
                })
                if budget_df.empty:
                    budget_df = new_record
                else:
                    budget_df = pd.concat([budget_df, new_record], ignore_index=True)
            
            # Save back to CSV
            budget_df.to_csv(self.budget_file, index=False)
            return True
            
        except Exception as e:
            print(f"Error setting category budget: {e}")
            return False
    
    def get_category_budget(self, year: int, month: int, category: str) -> float:
        """Get budget for a specific category in a month"""
        try:
            budget_df = self.get_all_budgets()
            
            if budget_df.empty:
                return 0.0
            
            # Filter for specific year, month, and category
            category_budget = budget_df[
                (budget_df['year'] == year) & 
                (budget_df['month'] == month) & 
                (budget_df['category'] == category)
            ]
            
            if not category_budget.empty:
                return float(category_budget.iloc[0]['budget_amount'])
            
            return 0.0
            
        except Exception as e:
            print(f"Error getting category budget: {e}")
            return 0.0
    
    def get_monthly_budgets(self, year: int, month: int) -> pd.DataFrame:
        """Get all budgets for a specific month"""
        try:
            budget_df = self.get_all_budgets()
            
            if budget_df.empty:
                return pd.DataFrame(columns=['year', 'month', 'category', 'budget_amount'])
            
            # Filter for specific year and month
            monthly_budgets = budget_df[
                (budget_df['year'] == year) & (budget_df['month'] == month)
            ]
            
            return monthly_budgets.reset_index(drop=True)
            
        except Exception as e:
            print(f"Error getting monthly budgets: {e}")
            return pd.DataFrame(columns=['year', 'month', 'category', 'budget_amount'])
    
    def get_all_budgets(self) -> pd.DataFrame:
        """Get all budget records"""
        try:
            if os.path.exists(self.budget_file) and os.path.getsize(self.budget_file) > 0:
                df = pd.read_csv(self.budget_file)
                if not df.empty:
                    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                    df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
                    df['budget_amount'] = pd.to_numeric(df['budget_amount'], errors='coerce')
                    
                    # Remove any rows with invalid data
                    df = df.dropna()
                    
                    return df
            
            return pd.DataFrame(columns=['year', 'month', 'category', 'budget_amount'])
            
        except Exception as e:
            print(f"Error reading budget data: {e}")
            return pd.DataFrame(columns=['year', 'month', 'category', 'budget_amount'])
    
    def get_budget_vs_actual(self, year: int, month: int) -> Dict[str, Dict[str, float]]:
        """Compare budgets vs actual spending for each category"""
        try:
            # Get expenses for the month
            expenses_df = self.get_expenses_by_month(year, month)
            actual_spending = {}
            if not expenses_df.empty:
                actual_spending = expenses_df.groupby('category')['amount'].sum().to_dict()
            
            # Get budgets for the month
            budgets_df = self.get_monthly_budgets(year, month)
            budget_amounts = {}
            if not budgets_df.empty:
                budget_amounts = budgets_df.set_index('category')['budget_amount'].to_dict()
            
            # Combine results
            result = {}
            all_categories = set(list(actual_spending.keys()) + list(budget_amounts.keys()))
            
            for category in all_categories:
                result[category] = {
                    'budget': budget_amounts.get(category, 0.0),
                    'actual': actual_spending.get(category, 0.0),
                    'remaining': budget_amounts.get(category, 0.0) - actual_spending.get(category, 0.0)
                }
            
            return result
            
        except Exception as e:
            print(f"Error comparing budget vs actual: {e}")
            return {}
