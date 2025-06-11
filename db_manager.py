import os
import pandas as pd
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, text, Column, Integer, String, Float, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()

class Expense(Base):
    __tablename__ = 'expenses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    description = Column(String(255), nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Income(Base):
    __tablename__ = 'income'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    income = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Budget(Base):
    __tablename__ = 'budgets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    category = Column(String(100), nullable=False)
    budget_amount = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSubscription(Base):
    __tablename__ = 'user_subscriptions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    amount = Column(Float, nullable=False)
    billing_cycle = Column(String(50), nullable=False)  # monthly, yearly, weekly
    next_billing_date = Column(Date, nullable=False)
    category = Column(String(100), nullable=False)
    description = Column(String(500))
    is_active = Column(String(10), default='active')  # active, paused, cancelled
    user_id = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    subscription_plan = Column(String(50), default='free')  # free, basic, premium
    subscription_status = Column(String(50), default='active')  # active, cancelled, expired
    subscription_start = Column(DateTime)
    subscription_end = Column(DateTime)
    payment_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class SubscriptionPlan(Base):
    __tablename__ = 'subscription_plans'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_name = Column(String(50), unique=True, nullable=False)
    display_name = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    billing_cycle = Column(String(20), nullable=False)  # monthly, yearly
    max_expenses = Column(Integer, default=-1)  # -1 for unlimited
    max_categories = Column(Integer, default=-1)
    has_predictions = Column(String(10), default='false')
    has_analytics = Column(String(10), default='false')
    has_budgets = Column(String(10), default='false')
    has_export = Column(String(10), default='false')
    description = Column(String(500))
    features = Column(String(1000))  # JSON string of features
    is_active = Column(String(10), default='true')
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
        except Exception as e:
            print(f"Error creating tables: {e}")
    
    def get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
    
    def add_expense(self, description: str, amount: float, category: str, expense_date: date) -> bool:
        """Add a new expense to the database"""
        try:
            with self.get_session() as session:
                expense = Expense(
                    date=expense_date,
                    description=description,
                    amount=amount,
                    category=category
                )
                session.add(expense)
                session.commit()
                return True
        except Exception as e:
            print(f"Error adding expense: {e}")
            return False
    
    def get_all_expenses(self) -> pd.DataFrame:
        """Get all expenses from the database"""
        try:
            with self.get_session() as session:
                query = text("SELECT id, date, description, amount, category FROM expenses ORDER BY date DESC")
                result = session.execute(query)
                
                expenses = []
                for row in result:
                    expenses.append({
                        'id': row.id,
                        'date': row.date,
                        'description': row.description,
                        'amount': row.amount,
                        'category': row.category
                    })
                
                if expenses:
                    df = pd.DataFrame(expenses)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                else:
                    return pd.DataFrame(columns=['id', 'date', 'description', 'amount', 'category'])
                    
        except Exception as e:
            print(f"Error reading expenses: {e}")
            return pd.DataFrame(columns=['id', 'date', 'description', 'amount', 'category'])
    
    def get_expenses_by_month(self, year: int, month: int) -> pd.DataFrame:
        """Get expenses for a specific month and year"""
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT id, date, description, amount, category 
                    FROM expenses 
                    WHERE EXTRACT(YEAR FROM date) = :year 
                    AND EXTRACT(MONTH FROM date) = :month
                    ORDER BY date DESC
                """)
                result = session.execute(query, {'year': year, 'month': month})
                
                expenses = []
                for row in result:
                    expenses.append({
                        'id': row.id,
                        'date': row.date,
                        'description': row.description,
                        'amount': row.amount,
                        'category': row.category
                    })
                
                if expenses:
                    df = pd.DataFrame(expenses)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                else:
                    return pd.DataFrame(columns=['id', 'date', 'description', 'amount', 'category'])
                    
        except Exception as e:
            print(f"Error reading monthly expenses: {e}")
            return pd.DataFrame(columns=['id', 'date', 'description', 'amount', 'category'])
    
    def set_monthly_income(self, year: int, month: int, income: float) -> bool:
        """Set or update monthly income"""
        try:
            with self.get_session() as session:
                # Check if record exists
                existing = session.query(Income).filter(
                    Income.year == year,
                    Income.month == month
                ).first()
                
                if existing:
                    existing.income = income
                else:
                    new_income = Income(year=year, month=month, income=income)
                    session.add(new_income)
                
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error setting monthly income: {e}")
            return False
    
    def get_monthly_income(self, year: int, month: int) -> float:
        """Get monthly income for a specific year and month"""
        try:
            with self.get_session() as session:
                income_record = session.query(Income).filter(
                    Income.year == year,
                    Income.month == month
                ).first()
                
                return float(income_record.income) if income_record else 0.0
                
        except Exception as e:
            print(f"Error getting monthly income: {e}")
            return 0.0
    
    def get_income_history(self) -> pd.DataFrame:
        """Get all income records"""
        try:
            with self.get_session() as session:
                query = text("SELECT year, month, income FROM income ORDER BY year DESC, month DESC")
                result = session.execute(query)
                
                income_records = []
                for row in result:
                    income_records.append({
                        'year': row.year,
                        'month': row.month,
                        'income': row.income
                    })
                
                if income_records:
                    return pd.DataFrame(income_records)
                else:
                    return pd.DataFrame(columns=['year', 'month', 'income'])
                    
        except Exception as e:
            print(f"Error reading income history: {e}")
            return pd.DataFrame(columns=['year', 'month', 'income'])
    
    def get_expense_categories(self) -> List[str]:
        """Get list of unique expense categories"""
        try:
            with self.get_session() as session:
                query = text("SELECT DISTINCT category FROM expenses ORDER BY category")
                result = session.execute(query)
                
                categories = [row.category for row in result]
                return categories
                
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
            with self.get_session() as session:
                # Check if record exists
                existing = session.query(Budget).filter(
                    Budget.year == year,
                    Budget.month == month,
                    Budget.category == category
                ).first()
                
                if existing:
                    existing.budget_amount = budget_amount
                else:
                    new_budget = Budget(
                        year=year,
                        month=month,
                        category=category,
                        budget_amount=budget_amount
                    )
                    session.add(new_budget)
                
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error setting category budget: {e}")
            return False
    
    def get_category_budget(self, year: int, month: int, category: str) -> float:
        """Get budget for a specific category in a month"""
        try:
            with self.get_session() as session:
                budget_record = session.query(Budget).filter(
                    Budget.year == year,
                    Budget.month == month,
                    Budget.category == category
                ).first()
                
                return float(budget_record.budget_amount) if budget_record else 0.0
                
        except Exception as e:
            print(f"Error getting category budget: {e}")
            return 0.0
    
    def get_monthly_budgets(self, year: int, month: int) -> pd.DataFrame:
        """Get all budgets for a specific month"""
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT year, month, category, budget_amount 
                    FROM budgets 
                    WHERE year = :year AND month = :month
                    ORDER BY category
                """)
                result = session.execute(query, {'year': year, 'month': month})
                
                budgets = []
                for row in result:
                    budgets.append({
                        'year': row.year,
                        'month': row.month,
                        'category': row.category,
                        'budget_amount': row.budget_amount
                    })
                
                if budgets:
                    return pd.DataFrame(budgets)
                else:
                    return pd.DataFrame(columns=['year', 'month', 'category', 'budget_amount'])
                    
        except Exception as e:
            print(f"Error getting monthly budgets: {e}")
            return pd.DataFrame(columns=['year', 'month', 'category', 'budget_amount'])
    
    def get_all_budgets(self) -> pd.DataFrame:
        """Get all budget records"""
        try:
            with self.get_session() as session:
                query = text("SELECT year, month, category, budget_amount FROM budgets ORDER BY year DESC, month DESC, category")
                result = session.execute(query)
                
                budgets = []
                for row in result:
                    budgets.append({
                        'year': row.year,
                        'month': row.month,
                        'category': row.category,
                        'budget_amount': row.budget_amount
                    })
                
                if budgets:
                    return pd.DataFrame(budgets)
                else:
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
    
    def create_user(self, user_id: str, email: str, name: str) -> bool:
        """Create a new user"""
        try:
            with self.get_session() as session:
                existing_user = session.query(User).filter(User.user_id == user_id).first()
                if existing_user:
                    return False
                
                new_user = User(
                    user_id=user_id,
                    email=email,
                    name=name,
                    subscription_plan='free',
                    subscription_status='active',
                    last_login=datetime.utcnow()
                )
                session.add(new_user)
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def get_user(self, user_id: str) -> dict:
        """Get user information"""
        try:
            with self.get_session() as session:
                user = session.query(User).filter(User.user_id == user_id).first()
                if user:
                    return {
                        'user_id': user.user_id,
                        'email': user.email,
                        'name': user.name,
                        'subscription_plan': user.subscription_plan,
                        'subscription_status': user.subscription_status,
                        'subscription_start': user.subscription_start,
                        'subscription_end': user.subscription_end,
                        'created_at': user.created_at,
                        'last_login': user.last_login
                    }
                return {}
                
        except Exception as e:
            print(f"Error getting user: {e}")
            return {}
    
    def update_user_subscription(self, user_id: str, plan: str, start_date: datetime, end_date: datetime, payment_id: str = None) -> bool:
        """Update user subscription plan"""
        try:
            with self.get_session() as session:
                user = session.query(User).filter(User.user_id == user_id).first()
                if user:
                    user.subscription_plan = plan
                    user.subscription_status = 'active'
                    user.subscription_start = start_date
                    user.subscription_end = end_date
                    if payment_id:
                        user.payment_id = payment_id
                    session.commit()
                    return True
                return False
                
        except Exception as e:
            print(f"Error updating user subscription: {e}")
            return False
    
    def initialize_subscription_plans(self) -> bool:
        """Initialize default subscription plans"""
        try:
            with self.get_session() as session:
                # Check if plans already exist
                existing_plans = session.query(SubscriptionPlan).count()
                if existing_plans > 0:
                    return True
                
                plans = [
                    {
                        'plan_name': 'free',
                        'display_name': 'Free Plan',
                        'price': 0.0,
                        'billing_cycle': 'monthly',
                        'max_expenses': 50,
                        'max_categories': 5,
                        'has_predictions': 'false',
                        'has_analytics': 'false',
                        'has_budgets': 'true',
                        'has_export': 'false',
                        'description': 'Basic expense tracking for personal use',
                        'features': '["Basic expense tracking", "5 categories", "50 expenses per month", "Basic budgets"]'
                    },
                    {
                        'plan_name': 'basic',
                        'display_name': 'Basic Plan',
                        'price': 9.99,
                        'billing_cycle': 'monthly',
                        'max_expenses': 500,
                        'max_categories': 15,
                        'has_predictions': 'false',
                        'has_analytics': 'true',
                        'has_budgets': 'true',
                        'has_export': 'true',
                        'description': 'Perfect for individuals and small families',
                        'features': '["Unlimited expenses", "15 categories", "Advanced analytics", "Budget management", "Data export", "Email support"]'
                    },
                    {
                        'plan_name': 'premium',
                        'display_name': 'Premium Plan',
                        'price': 19.99,
                        'billing_cycle': 'monthly',
                        'max_expenses': -1,
                        'max_categories': -1,
                        'has_predictions': 'true',
                        'has_analytics': 'true',
                        'has_budgets': 'true',
                        'has_export': 'true',
                        'description': 'Complete financial management with AI predictions',
                        'features': '["Unlimited everything", "AI predictions", "Advanced analytics", "Premium support", "Custom categories", "Data export", "Priority features"]'
                    }
                ]
                
                for plan_data in plans:
                    plan = SubscriptionPlan(**plan_data)
                    session.add(plan)
                
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error initializing subscription plans: {e}")
            return False
    
    def get_subscription_plans(self) -> pd.DataFrame:
        """Get all active subscription plans"""
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT plan_name, display_name, price, billing_cycle, 
                           max_expenses, max_categories, has_predictions, 
                           has_analytics, has_budgets, has_export, 
                           description, features
                    FROM subscription_plans 
                    WHERE is_active = 'true'
                    ORDER BY price ASC
                """)
                result = session.execute(query)
                
                plans = []
                for row in result:
                    plans.append({
                        'plan_name': row.plan_name,
                        'display_name': row.display_name,
                        'price': row.price,
                        'billing_cycle': row.billing_cycle,
                        'max_expenses': row.max_expenses,
                        'max_categories': row.max_categories,
                        'has_predictions': row.has_predictions,
                        'has_analytics': row.has_analytics,
                        'has_budgets': row.has_budgets,
                        'has_export': row.has_export,
                        'description': row.description,
                        'features': row.features
                    })
                
                if plans:
                    return pd.DataFrame(plans)
                else:
                    return pd.DataFrame(columns=['plan_name', 'display_name', 'price', 'billing_cycle', 'max_expenses', 'max_categories', 'has_predictions', 'has_analytics', 'has_budgets', 'has_export', 'description', 'features'])
                    
        except Exception as e:
            print(f"Error reading subscription plans: {e}")
            return pd.DataFrame(columns=['plan_name', 'display_name', 'price', 'billing_cycle', 'max_expenses', 'max_categories', 'has_predictions', 'has_analytics', 'has_budgets', 'has_export', 'description', 'features'])
    
    def check_user_limits(self, user_id: str) -> dict:
        """Check user's current usage against their plan limits"""
        try:
            user = self.get_user(user_id)
            if not user:
                return {'valid': False, 'message': 'User not found'}
            
            # Get user's plan
            plans_df = self.get_subscription_plans()
            user_plan = plans_df[plans_df['plan_name'] == user['subscription_plan']]
            
            if user_plan.empty:
                return {'valid': False, 'message': 'Invalid subscription plan'}
            
            plan = user_plan.iloc[0]
            
            # Check expense limit
            current_month_expenses = len(self.get_expenses_by_month(
                datetime.now().year, datetime.now().month
            ))
            
            # Check category limit
            all_expenses = self.get_all_expenses()
            current_categories = len(all_expenses['category'].unique()) if not all_expenses.empty else 0
            
            limits = {
                'valid': True,
                'plan_name': plan['display_name'],
                'expenses_used': current_month_expenses,
                'expenses_limit': plan['max_expenses'],
                'categories_used': current_categories,
                'categories_limit': plan['max_categories'],
                'can_add_expense': plan['max_expenses'] == -1 or current_month_expenses < plan['max_expenses'],
                'can_add_category': plan['max_categories'] == -1 or current_categories < plan['max_categories'],
                'has_predictions': plan['has_predictions'] == 'true',
                'has_analytics': plan['has_analytics'] == 'true',
                'has_budgets': plan['has_budgets'] == 'true',
                'has_export': plan['has_export'] == 'true'
            }
            
            return limits
                
        except Exception as e:
            print(f"Error checking user limits: {e}")
            return {'valid': False, 'message': 'Error checking limits'}

    def add_user_subscription(self, user_id: str, name: str, amount: float, billing_cycle: str, next_billing_date: date, category: str, description: str = "") -> bool:
        """Add a new subscription"""
        try:
            with self.get_session() as session:
                new_subscription = UserSubscription(
                    name=name,
                    amount=amount,
                    billing_cycle=billing_cycle,
                    next_billing_date=next_billing_date,
                    category=category,
                    description=description,
                    is_active='active',
                    user_id=user_id
                )
                session.add(new_subscription)
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error adding subscription: {e}")
            return False
    
    def get_all_user_subscriptions(self, user_id: str) -> pd.DataFrame:
        """Get all subscriptions"""
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT id, name, amount, billing_cycle, next_billing_date, 
                           category, description, is_active, created_at 
                    FROM subscriptions 
                    ORDER BY name
                """)
                result = session.execute(query)
                
                subscriptions = []
                for row in result:
                    subscriptions.append({
                        'id': row.id,
                        'name': row.name,
                        'amount': row.amount,
                        'billing_cycle': row.billing_cycle,
                        'next_billing_date': row.next_billing_date,
                        'category': row.category,
                        'description': row.description,
                        'is_active': row.is_active,
                        'created_at': row.created_at
                    })
                
                if subscriptions:
                    df = pd.DataFrame(subscriptions)
                    df['next_billing_date'] = pd.to_datetime(df['next_billing_date']).dt.date
                    return df
                else:
                    return pd.DataFrame(columns=['id', 'name', 'amount', 'billing_cycle', 'next_billing_date', 'category', 'description', 'is_active', 'created_at'])
                    
        except Exception as e:
            print(f"Error reading subscriptions: {e}")
            return pd.DataFrame(columns=['id', 'name', 'amount', 'billing_cycle', 'next_billing_date', 'category', 'description', 'is_active', 'created_at'])
    
    def get_active_subscriptions(self) -> pd.DataFrame:
        """Get only active subscriptions"""
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT id, name, amount, billing_cycle, next_billing_date, 
                           category, description, is_active, created_at 
                    FROM subscriptions 
                    WHERE is_active = 'active'
                    ORDER BY next_billing_date ASC
                """)
                result = session.execute(query)
                
                subscriptions = []
                for row in result:
                    subscriptions.append({
                        'id': row.id,
                        'name': row.name,
                        'amount': row.amount,
                        'billing_cycle': row.billing_cycle,
                        'next_billing_date': row.next_billing_date,
                        'category': row.category,
                        'description': row.description,
                        'is_active': row.is_active,
                        'created_at': row.created_at
                    })
                
                if subscriptions:
                    df = pd.DataFrame(subscriptions)
                    df['next_billing_date'] = pd.to_datetime(df['next_billing_date']).dt.date
                    return df
                else:
                    return pd.DataFrame(columns=['id', 'name', 'amount', 'billing_cycle', 'next_billing_date', 'category', 'description', 'is_active', 'created_at'])
                    
        except Exception as e:
            print(f"Error reading active subscriptions: {e}")
            return pd.DataFrame(columns=['id', 'name', 'amount', 'billing_cycle', 'next_billing_date', 'category', 'description', 'is_active', 'created_at'])
    
    def update_subscription_status(self, subscription_id: int, status: str) -> bool:
        """Update subscription status (active, paused, cancelled)"""
        try:
            with self.get_session() as session:
                subscription = session.query(Subscription).filter(Subscription.id == subscription_id).first()
                if subscription:
                    subscription.is_active = status
                    session.commit()
                    return True
                return False
                
        except Exception as e:
            print(f"Error updating subscription status: {e}")
            return False
    
    def update_subscription_next_billing(self, subscription_id: int, next_date: date) -> bool:
        """Update subscription next billing date"""
        try:
            with self.get_session() as session:
                subscription = session.query(Subscription).filter(Subscription.id == subscription_id).first()
                if subscription:
                    subscription.next_billing_date = next_date
                    session.commit()
                    return True
                return False
                
        except Exception as e:
            print(f"Error updating subscription billing date: {e}")
            return False
    
    def get_upcoming_subscriptions(self, days_ahead: int = 7) -> pd.DataFrame:
        """Get subscriptions due in the next X days"""
        try:
            from datetime import timedelta
            
            today = date.today()
            future_date = today + timedelta(days=days_ahead)
            
            with self.get_session() as session:
                query = text("""
                    SELECT id, name, amount, billing_cycle, next_billing_date, 
                           category, description, is_active, created_at 
                    FROM subscriptions 
                    WHERE is_active = 'active' 
                    AND next_billing_date BETWEEN :today AND :future_date
                    ORDER BY next_billing_date ASC
                """)
                result = session.execute(query, {'today': today, 'future_date': future_date})
                
                subscriptions = []
                for row in result:
                    subscriptions.append({
                        'id': row.id,
                        'name': row.name,
                        'amount': row.amount,
                        'billing_cycle': row.billing_cycle,
                        'next_billing_date': row.next_billing_date,
                        'category': row.category,
                        'description': row.description,
                        'is_active': row.is_active,
                        'created_at': row.created_at
                    })
                
                if subscriptions:
                    df = pd.DataFrame(subscriptions)
                    df['next_billing_date'] = pd.to_datetime(df['next_billing_date']).dt.date
                    return df
                else:
                    return pd.DataFrame(columns=['id', 'name', 'amount', 'billing_cycle', 'next_billing_date', 'category', 'description', 'is_active', 'created_at'])
                    
        except Exception as e:
            print(f"Error getting upcoming subscriptions: {e}")
            return pd.DataFrame(columns=['id', 'name', 'amount', 'billing_cycle', 'next_billing_date', 'category', 'description', 'is_active', 'created_at'])
    
    def get_monthly_subscription_cost(self, year: int, month: int) -> float:
        """Calculate total subscription cost for a specific month"""
        try:
            active_subs = self.get_active_subscriptions()
            if active_subs.empty:
                return 0.0
            
            total_cost = 0.0
            
            for _, sub in active_subs.iterrows():
                if sub['billing_cycle'] == 'monthly':
                    total_cost += sub['amount']
                elif sub['billing_cycle'] == 'yearly':
                    total_cost += sub['amount'] / 12
                elif sub['billing_cycle'] == 'weekly':
                    total_cost += sub['amount'] * 4.33  # Average weeks per month
            
            return total_cost
            
        except Exception as e:
            print(f"Error calculating monthly subscription cost: {e}")
            return 0.0
    
    def delete_subscription(self, subscription_id: int) -> bool:
        """Delete a subscription"""
        try:
            with self.get_session() as session:
                subscription = session.query(Subscription).filter(Subscription.id == subscription_id).first()
                if subscription:
                    session.delete(subscription)
                    session.commit()
                    return True
                return False
                
        except Exception as e:
            print(f"Error deleting subscription: {e}")
            return False