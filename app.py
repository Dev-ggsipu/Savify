import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from db_manager import DatabaseManager
from auth_manager import AuthManager
import calendar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Initialize managers
@st.cache_resource
def init_data_manager():
    return DatabaseManager()

@st.cache_resource
def init_auth_manager():
    return AuthManager()

def apply_custom_css():
    """Apply custom CSS for better styling"""
    st.markdown("""
    <style>
    .section-header {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-header h2 {
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86AB;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Savify = just save it !!!", page_icon="💰", layout="wide")
    
    apply_custom_css()
    
    # Initialize managers
    data_manager = init_data_manager()
    auth_manager = init_auth_manager()
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = {}
    
    # Check authentication
    if not st.session_state.authenticated:
        auth_manager.show_login_page()
        return
    
    # Get current user
    user = st.session_state.user
    user_id = user.get('user_id', '')
    
    # Check user limits for navigation restrictions
    limits = data_manager.check_user_limits(user_id)
    
    # Sidebar for navigation
    with st.sidebar:
        # User info header
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e9ecef; margin-bottom: 1rem;">
            <h2 style="color: #2E86AB; margin-bottom: 0.5rem;">💰 Savify = just save it !!!</h2>
            <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">Welcome, {user.get('name', 'User')}</p>
            <p style="color: #28a745; font-size: 0.8rem; font-weight: bold; margin: 0.2rem 0 0 0;">{limits.get('plan_name', 'Free Plan')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation options with feature restrictions
        nav_options = []
        nav_options.append(("🏠 Dashboard", "Dashboard"))
        nav_options.append(("➕ Add Expense", "Add Expense"))
        nav_options.append(("💼 Manage Income", "Manage Income"))
        nav_options.append(("📋 View Expenses", "View Expenses"))
        
        if limits.get('has_budgets', True):  # Free plan has basic budgets
            nav_options.append(("💰 Budget Management", "Budget Management"))
        
        if limits.get('has_analytics', False):
            nav_options.append(("📊 Charts & Analytics", "Charts & Analytics"))
        
        if limits.get('has_predictions', False):
            nav_options.append(("🔮 Future Predictions", "Future Predictions"))
        
        nav_options.append(("💎 Upgrade Plan", "Upgrade Plan"))
        
        # Create a mapping for the selectbox
        option_mapping = {option[1]: option[0] for option in nav_options}
        
        page = st.selectbox(
            "Select Page:",
            options=[option[1] for option in nav_options],
            format_func=lambda x: option_mapping.get(x, x),
            index=0
        )
        
        # Usage statistics
        st.markdown("### Usage This Month")
        expenses_used = limits.get('expenses_used', 0)
        expenses_limit = limits.get('expenses_limit', 50)
        
        if expenses_limit == -1:
            usage_text = f"Expenses: {expenses_used} (Unlimited)"
        else:
            usage_text = f"Expenses: {expenses_used}/{expenses_limit}"
            if expenses_used >= expenses_limit:
                st.error("Expense limit reached!")
            elif expenses_used >= expenses_limit * 0.8:
                st.warning("Approaching expense limit")
        
        st.info(usage_text)
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = {}
            st.rerun()
    
    # Route to appropriate page with user context
    if page == "Dashboard":
        show_dashboard(data_manager)
    elif page == "Add Expense":
        add_expense_page(data_manager, user_id, limits)
    elif page == "Manage Income":
        manage_income_page(data_manager)
    elif page == "View Expenses":
        view_expenses_page(data_manager)
    elif page == "Budget Management":
        if limits.get('has_budgets', True):
            budget_management_page(data_manager)
        else:
            show_upgrade_required("Budget Management", auth_manager)
    elif page == "Charts & Analytics":
        if limits.get('has_analytics', False):
            charts_analytics_page(data_manager)
        else:
            show_upgrade_required("Charts & Analytics", auth_manager)
    elif page == "Future Predictions":
        if limits.get('has_predictions', False):
            future_predictions_page(data_manager)
        else:
            show_upgrade_required("Future Predictions", auth_manager)
    elif page == "Upgrade Plan":
        auth_manager.show_subscription_page()

def show_upgrade_required(feature_name, auth_manager):
    """Show upgrade required message"""
    st.markdown(f'<div class="section-header"><h2>🔒 {feature_name}</h2></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 2rem; background: #f8f9fa; border-radius: 12px; border: 2px solid #e9ecef;">
        <h3 style="color: #6c757d; margin-bottom: 1rem;">🔒 Premium Feature</h3>
        <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;">
            {feature_name} is available with our paid plans.<br>
            Upgrade to unlock this feature and many more!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 View Upgrade Options", use_container_width=True):
            st.session_state.page = "Upgrade Plan"
            st.rerun()

def add_expense_page(data_manager, user_id, limits):
    """Add expense page with usage limits"""
    st.markdown('<div class="section-header"><h2>➕ Add New Expense</h2></div>', unsafe_allow_html=True)
    
    # Check usage limits
    expenses_used = limits.get('expenses_used', 0)
    expenses_limit = limits.get('expenses_limit', 50)
    
    if expenses_limit != -1 and expenses_used >= expenses_limit:
        st.error(f"You've reached your monthly limit of {expenses_limit} expenses. Upgrade your plan to add more expenses.")
        return
    
    with st.form("add_expense"):
        col1, col2 = st.columns(2)
        
        with col1:
            description = st.text_input("Description", placeholder="Coffee, Groceries, etc.")
            amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)
        
        with col2:
            # Get existing categories but limit for free users
            existing_categories = data_manager.get_expense_categories()
            categories_limit = limits.get('categories_limit', 5)
            
            if len(existing_categories) >= categories_limit and categories_limit != -1:
                st.info(f"Category limit reached ({categories_limit}). Choose from existing categories or upgrade your plan.")
                category = st.selectbox("Category", existing_categories)
            else:
                all_categories = ["Food & Dining", "Transportation", "Shopping", "Entertainment", 
                                "Bills & Utilities", "Healthcare", "Education", "Travel", "Other"]
                combined_categories = list(set(existing_categories + all_categories))
                category = st.selectbox("Category", combined_categories)
            
            expense_date = st.date_input("Date", value=date.today())
        
        submitted = st.form_submit_button("Add Expense", use_container_width=True)
        
        if submitted and description and amount:
            success = data_manager.add_expense(description, amount, category, expense_date)
            if success:
                st.success(f"Added expense: {description} - ${amount:.2f}")
                st.rerun()
            else:
                st.error("Failed to add expense")

def show_dashboard(data_manager):
    st.markdown('<div class="section-header"><h2>📊 Financial Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Current month summary
    current_date = datetime.now()
    summary = data_manager.get_monthly_summary(current_date.year, current_date.month)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Monthly Income",
            value=f"${summary['income']:.2f}"
        )
    
    with col2:
        st.metric(
            label="💸 Total Expenses",
            value=f"${summary['total_expenses']:.2f}"
        )
    
    with col3:
        remaining = summary['income'] - summary['total_expenses']
        st.metric(
            label="💵 Remaining",
            value=f"${remaining:.2f}",
            delta=f"${remaining:.2f}" if remaining >= 0 else f"-${abs(remaining):.2f}"
        )
    
    with col4:
        st.metric(
            label="📝 Total Transactions",
            value=summary['expense_count']
        )
    
    # Recent expenses preview
    st.subheader("📋 Recent Expenses")
    recent_expenses = data_manager.get_expenses_by_month(current_date.year, current_date.month)
    
    if not recent_expenses.empty:
        # Show last 5 expenses
        recent_expenses = recent_expenses.sort_values('date', ascending=False).head(5)
        st.dataframe(recent_expenses[['date', 'description', 'category', 'amount']], use_container_width=True)
    else:
        st.info("No expenses recorded this month. Start by adding your first expense!")

def manage_income_page(data_manager):
    st.markdown('<div class="section-header"><h2>💼 Manage Monthly Income</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Set Income")
        
        with st.form("set_income"):
            year = st.number_input("Year", min_value=2020, max_value=2030, value=datetime.now().year)
            month_names = [calendar.month_name[i] for i in range(1, 13)]
            selected_month_name = st.selectbox("Month", month_names)
            month = month_names.index(selected_month_name) + 1
            income = st.number_input("Monthly Income ($)", min_value=0.0, step=100.0)
            
            submitted = st.form_submit_button("Set Income")
            
            if submitted and income >= 0:
                success = data_manager.set_monthly_income(year, month, income)
                if success:
                    st.success(f"Set income for {calendar.month_name[month]} {year}: ${income:.2f}")
                    st.rerun()
                else:
                    st.error("Failed to set income")
    
    with col2:
        st.subheader("Income History")
        
        income_history = data_manager.get_income_history()
        
        if not income_history.empty:
            # Add month names for better readability
            income_history['month_name'] = income_history['month'].apply(lambda x: calendar.month_name[x])
            income_history = income_history.sort_values(['year', 'month'], ascending=False)
            
            # Display formatted table
            display_df = income_history[['year', 'month_name', 'income']].copy()
            display_df.columns = ['Year', 'Month', 'Income ($)']
            display_df['Income ($)'] = display_df['Income ($)'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No income history available. Set your first monthly income above.")

def view_expenses_page(data_manager):
    st.markdown('<div class="section-header"><h2>📋 View & Manage Expenses</h2></div>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year = st.selectbox("Year", range(2020, 2031), index=range(2020, 2031).index(datetime.now().year))
    
    with col2:
        months = [(i, calendar.month_name[i]) for i in range(1, 13)]
        month_names = [name for _, name in months]
        selected_month_name = st.selectbox("Month", month_names, index=datetime.now().month - 1)
        month = next(num for num, name in months if name == selected_month_name)
    
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Get and display expenses
    expenses = data_manager.get_expenses_by_month(year, month)
    
    if not expenses.empty:
        # Summary for the month
        total_amount = expenses['amount'].sum()
        avg_amount = expenses['amount'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Spent", f"${total_amount:.2f}")
        with col2:
            st.metric("Average per Transaction", f"${avg_amount:.2f}")
        with col3:
            st.metric("Number of Transactions", len(expenses))
        
        # Category filter
        categories = ['All'] + list(expenses['category'].unique())
        selected_category = st.selectbox("Filter by Category", categories)
        
        if selected_category != 'All':
            expenses = expenses[expenses['category'] == selected_category]
        
        # Sort options
        sort_by = st.selectbox("Sort by", ['Date (Latest)', 'Date (Oldest)', 'Amount (High)', 'Amount (Low)'])
        
        if sort_by == 'Date (Latest)':
            expenses = expenses.sort_values('date', ascending=False)
        elif sort_by == 'Date (Oldest)':
            expenses = expenses.sort_values('date', ascending=True)
        elif sort_by == 'Amount (High)':
            expenses = expenses.sort_values('amount', ascending=False)
        elif sort_by == 'Amount (Low)':
            expenses = expenses.sort_values('amount', ascending=True)
        
        # Display expenses table
        st.subheader(f"Expenses for {calendar.month_name[month]} {year}")
        
        # Format the dataframe for better display
        display_df = expenses.copy()
        display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Category breakdown
        if len(expenses) > 0:
            st.subheader("Category Breakdown")
            category_summary = expenses.groupby('category')['amount'].agg(['sum', 'count']).round(2)
            category_summary.columns = ['Total ($)', 'Count']
            category_summary = category_summary.sort_values('Total ($)', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(category_summary, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(values=category_summary['Total ($)'], 
                           names=category_summary.index,
                           title="Spending by Category")
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info(f"No expenses found for {calendar.month_name[month]} {year}. Start adding some expenses!")

def budget_management_page(data_manager):
    st.markdown('<div class="section-header"><h2>💰 Budget Management</h2></div>', unsafe_allow_html=True)
    
    # Current month budget setup
    current_date = datetime.now()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Set Category Budget")
        
        with st.form("set_budget"):
            year = st.number_input("Year", min_value=2020, max_value=2030, value=current_date.year)
            month_names = [calendar.month_name[i] for i in range(1, 13)]
            selected_month_name = st.selectbox("Month", month_names, index=current_date.month - 1)
            month = month_names.index(selected_month_name) + 1
            
            # Get existing categories
            categories = data_manager.get_expense_categories()
            if not categories:
                categories = ["Food & Dining", "Transportation", "Shopping", "Entertainment", 
                            "Bills & Utilities", "Healthcare", "Education", "Travel", "Other"]
            
            category = st.selectbox("Category", categories)
            budget_amount = st.number_input("Budget Amount ($)", min_value=0.0, step=50.0)
            
            submitted = st.form_submit_button("Set Budget")
            
            if submitted and budget_amount > 0:
                success = data_manager.set_category_budget(year, month, category, budget_amount)
                if success:
                    st.success(f"Set budget for {category}: ${budget_amount:.2f}")
                    st.rerun()
                else:
                    st.error("Failed to set budget")
    
    with col2:
        st.subheader(f"Budget vs Actual - {calendar.month_name[current_date.month]} {current_date.year}")
        
        # Get budget vs actual comparison
        comparison = data_manager.get_budget_vs_actual(current_date.year, current_date.month)
        
        if comparison:
            budget_data = []
            
            for category, data in comparison.items():
                budget_data.append({
                    'Category': category,
                    'Budget': data['budget'],
                    'Actual': data['actual'],
                    'Remaining': data['budget'] - data['actual'],
                    'Status': '✅ Under' if data['actual'] <= data['budget'] else '⚠️ Over'
                })
            
            if budget_data:
                budget_df = pd.DataFrame(budget_data)
                
                # Format currency columns
                for col in ['Budget', 'Actual', 'Remaining']:
                    budget_df[f'{col}_formatted'] = budget_df[col].apply(lambda x: f"${x:.2f}")
                
                # Display table
                display_df = budget_df[['Category', 'Budget_formatted', 'Actual_formatted', 'Remaining_formatted', 'Status']].copy()
                display_df.columns = ['Category', 'Budget', 'Actual', 'Remaining', 'Status']
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Visual comparison
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Budget',
                    x=budget_df['Category'],
                    y=budget_df['Budget'],
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name='Actual',
                    x=budget_df['Category'],
                    y=budget_df['Actual'],
                    marker_color='lightcoral'
                ))
                
                fig.update_layout(
                    title="Budget vs Actual Spending",
                    xaxis_title="Category",
                    yaxis_title="Amount ($)",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No budgets set for this month. Set your first budget using the form on the left.")
        else:
            st.info("No budgets set for this month. Set your first budget using the form on the left.")

def charts_analytics_page(data_manager):
    st.markdown('<div class="section-header"><h2>📊 Charts & Analytics</h2></div>', unsafe_allow_html=True)
    
    # Get all expenses for analysis
    all_expenses = data_manager.get_all_expenses()
    
    if all_expenses.empty:
        st.info("No expense data available for analysis. Start by adding some expenses!")
        return
    
    # Convert date column to datetime
    all_expenses['date'] = pd.to_datetime(all_expenses['date'])
    all_expenses['year_month'] = all_expenses['date'].dt.to_period('M')
    
    # Monthly trend analysis
    st.subheader("📈 Monthly Spending Trends")
    
    monthly_totals = all_expenses.groupby('year_month')['amount'].sum().reset_index()
    monthly_totals['year_month_str'] = monthly_totals['year_month'].astype(str)
    
    fig_trend = px.line(monthly_totals, x='year_month_str', y='amount',
                       title="Monthly Spending Trend",
                       labels={'year_month_str': 'Month', 'amount': 'Total Spending ($)'})
    fig_trend.update_traces(line_color='#2E86AB', line_width=3)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Category analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💳 Spending by Category")
        category_totals = all_expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        fig_cat = px.pie(values=category_totals.values, names=category_totals.index,
                        title="Total Spending Distribution")
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.subheader("📊 Category Breakdown")
        category_stats = all_expenses.groupby('category')['amount'].agg(['sum', 'mean', 'count']).round(2)
        category_stats.columns = ['Total ($)', 'Average ($)', 'Count']
        category_stats = category_stats.sort_values('Total ($)', ascending=False)
        st.dataframe(category_stats, use_container_width=True)
    
    # Time-based analysis
    st.subheader("📅 Spending Patterns")
    
    # Day of week analysis
    all_expenses['day_of_week'] = all_expenses['date'].dt.day_name()
    daily_spending = all_expenses.groupby('day_of_week')['amount'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    fig_daily = px.bar(x=daily_spending.index, y=daily_spending.values,
                      title="Average Spending by Day of Week",
                      labels={'x': 'Day of Week', 'y': 'Average Amount ($)'})
    fig_daily.update_traces(marker_color='#A23B72')
    st.plotly_chart(fig_daily, use_container_width=True)

def future_predictions_page(data_manager):
    st.markdown('<div class="section-header"><h2>🔮 Future Expense Predictions</h2></div>', unsafe_allow_html=True)
    
    # Get expense data for predictions
    expenses_df = data_manager.get_all_expenses()
    
    if expenses_df.empty:
        st.info("No expense data available for predictions. Add some expenses first!")
        return
    
    # Prepare data for prediction
    monthly_data = prepare_prediction_data(expenses_df)
    
    if len(monthly_data) < 3:
        st.warning("Need at least 3 months of data for reliable predictions. Keep tracking your expenses!")
        return
    
    # Prediction controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Prediction Settings")
        prediction_months = st.slider("Months to Predict", 1, 12, 6)
        
        if st.button("Generate Predictions"):
            st.session_state.predictions_generated = True
    
    with col2:
        if st.session_state.get('predictions_generated', False):
            # Generate and display predictions
            predictions = predict_total_expenses(monthly_data, prediction_months)
            display_total_predictions(predictions, monthly_data)
    
    # Category-wise predictions
    if st.session_state.get('predictions_generated', False):
        st.subheader("📊 Category-wise Predictions")
        category_predictions = predict_category_expenses(expenses_df, prediction_months)
        display_category_predictions(category_predictions, expenses_df)
        
        # Insights and recommendations
        display_prediction_insights(monthly_data, expenses_df)

def prepare_prediction_data(expenses_df):
    """Prepare monthly aggregated data for prediction"""
    expenses_df['date'] = pd.to_datetime(expenses_df['date'])
    expenses_df['year_month'] = expenses_df['date'].dt.to_period('M')
    
    monthly_data = expenses_df.groupby('year_month')['amount'].sum().reset_index()
    monthly_data['month_num'] = range(len(monthly_data))
    
    return monthly_data

def predict_total_expenses(monthly_data, prediction_months):
    """Predict total monthly expenses using polynomial regression"""
    if len(monthly_data) < 3:
        return None
    
    X = monthly_data[['month_num']]
    y = monthly_data['amount']
    
    # Use polynomial features for better fit
    poly_degree = min(2, len(monthly_data) - 1)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_degree)),
        ('linear', LinearRegression())
    ])
    
    model.fit(X, y)
    
    # Generate future predictions
    future_months = range(len(monthly_data), len(monthly_data) + prediction_months)
    future_X = pd.DataFrame({'month_num': future_months})
    predictions = model.predict(future_X)
    
    # Ensure predictions are not negative
    predictions = np.maximum(predictions, 0)
    
    return {
        'model': model,
        'predictions': predictions,
        'future_months': future_months,
        'accuracy': model.score(X, y)
    }

def display_total_predictions(predictions, historical_data):
    """Display total expense predictions with charts"""
    if predictions is None:
        st.error("Unable to generate predictions with current data")
        return
    
    # Create future date labels
    last_period = historical_data['year_month'].iloc[-1]
    future_periods = []
    
    for i in range(len(predictions['predictions'])):
        future_period = last_period + i + 1
        future_periods.append(str(future_period))
    
    # Combine historical and predicted data for visualization
    historical_amounts = historical_data['amount'].tolist()
    predicted_amounts = predictions['predictions'].tolist()
    
    historical_periods = [str(p) for p in historical_data['year_month']]
    
    # Create the prediction chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_periods,
        y=historical_amounts,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#2E86AB', width=3)
    ))
    
    # Predicted data
    fig.add_trace(go.Scatter(
        x=future_periods,
        y=predicted_amounts,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#A23B72', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title="Expense Prediction Forecast",
        xaxis_title="Month",
        yaxis_title="Total Expenses ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display prediction summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_prediction = np.mean(predicted_amounts)
        st.metric("Avg Predicted Monthly", f"${avg_prediction:.2f}")
    
    with col2:
        total_prediction = np.sum(predicted_amounts)
        st.metric("Total Predicted", f"${total_prediction:.2f}")
    
    with col3:
        model_accuracy = predictions['accuracy'] * 100
        st.metric("Model Accuracy", f"{model_accuracy:.1f}%")

def predict_category_expenses(expenses_df, prediction_months):
    """Predict expenses by category"""
    expenses_df['date'] = pd.to_datetime(expenses_df['date'])
    expenses_df['year_month'] = expenses_df['date'].dt.to_period('M')
    
    category_predictions = {}
    
    for category in expenses_df['category'].unique():
        category_data = expenses_df[expenses_df['category'] == category]
        monthly_category = category_data.groupby('year_month')['amount'].sum().reset_index()
        
        if len(monthly_category) >= 2:
            monthly_category['month_num'] = range(len(monthly_category))
            
            X = monthly_category[['month_num']]
            y = monthly_category['amount']
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_months = range(len(monthly_category), len(monthly_category) + prediction_months)
            future_X = pd.DataFrame({'month_num': future_months})
            predictions = model.predict(future_X)
            predictions = np.maximum(predictions, 0)  # No negative predictions
            
            category_predictions[category] = {
                'predictions': predictions,
                'avg_prediction': np.mean(predictions),
                'total_prediction': np.sum(predictions)
            }
    
    return category_predictions

def display_category_predictions(category_predictions, expenses_df):
    """Display category-wise predictions"""
    if not category_predictions:
        st.info("Not enough data for category-wise predictions")
        return
    
    # Create summary table
    summary_data = []
    for category, data in category_predictions.items():
        summary_data.append({
            'Category': category,
            'Avg Monthly': f"${data['avg_prediction']:.2f}",
            'Total Predicted': f"${data['total_prediction']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Category prediction chart
    categories = list(category_predictions.keys())
    avg_predictions = [data['avg_prediction'] for data in category_predictions.values()]
    
    fig = px.bar(x=categories, y=avg_predictions,
                title="Predicted Average Monthly Spending by Category",
                labels={'x': 'Category', 'y': 'Predicted Amount ($)'})
    fig.update_traces(marker_color='#2E86AB')
    st.plotly_chart(fig, use_container_width=True)

def display_prediction_insights(monthly_data, expenses_df):
    """Display insights about prediction model and spending patterns"""
    st.subheader("💡 Prediction Insights")
    
    # Calculate trends
    if len(monthly_data) >= 6:
        recent_avg = monthly_data.tail(3)['amount'].mean()
        earlier_avg = monthly_data.head(3)['amount'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Recent 3-Month Avg", f"${recent_avg:.2f}")
        
        with col2:
            st.metric("Earlier 3-Month Avg", f"${earlier_avg:.2f}")
        
        with col3:
            if recent_avg > earlier_avg * 1.1:
                trend = "📈 Increasing"
            elif recent_avg < earlier_avg * 0.9:
                trend = "📉 Decreasing"
            else:
                trend = "➡️ Stable"
            
            st.metric("Overall Trend", trend)
    
    # Recommendations
    st.markdown("#### 💡 Recommendations")
    
    if len(monthly_data) < 6:
        st.info("💡 Add more expense data to improve prediction accuracy. Aim for at least 6 months of consistent tracking.")
    
    if len(monthly_data) > 0:
        avg_monthly = monthly_data['amount'].mean()
        std_monthly = monthly_data['amount'].std()
        
        if std_monthly / avg_monthly > 0.3:
            st.warning("💡 Your spending varies significantly month-to-month. Consider setting consistent monthly budgets to improve financial stability.")
        
        # Category recommendations
        if not expenses_df.empty:
            category_spending = expenses_df.groupby('category')['amount'].sum().sort_values(ascending=False)
            top_category = category_spending.index[0]
            top_amount = category_spending.iloc[0]
            
            if top_amount > avg_monthly * 0.4:
                st.info(f"💡 {top_category} represents a large portion of your spending (${top_amount:,.2f}). Consider reviewing this category for potential savings.")

if __name__ == "__main__":
    main()