# Savify = just save it !!!

A comprehensive expense tracking application built with Streamlit, featuring user authentication, subscription management, and advanced analytics.

## Features

### Core Functionality
- **Expense Tracking**: Add, view, and categorize expenses
- **Income Management**: Set and track monthly income
- **Budget Management**: Set category budgets and track spending
- **Dashboard**: Overview of financial status with key metrics

### Premium Features
- **Charts & Analytics**: Interactive visualizations and spending analysis
- **Future Predictions**: AI-powered expense forecasting using machine learning
- **Advanced Reports**: Detailed financial insights and trends

### Subscription System
- **Free Plan**: Basic expense tracking (50 expenses/month, 5 categories)
- **Basic Plan ($9.99/month)**: Enhanced features with budget management
- **Premium Plan ($19.99/month)**: Full access including AI predictions and analytics

## Technology Stack

- **Frontend**: Streamlit
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: Custom user management system
- **Visualization**: Plotly for interactive charts
- **Machine Learning**: Scikit-learn for expense predictions
- **Styling**: Custom CSS for professional UI

## Installation

1. Install dependencies:
```bash
pip install streamlit pandas plotly scikit-learn sqlalchemy psycopg2-binary numpy
```

2. Set up PostgreSQL database and configure DATABASE_URL environment variable

3. Run the application:
```bash
streamlit run app.py --server.port 5000
```

## File Structure

- `app.py` - Main application file with UI and routing
- `auth_manager.py` - User authentication and subscription management
- `db_manager.py` - Database operations and models
- `data_manager.py` - CSV-based data operations (legacy)
- `pyproject.toml` - Project dependencies

## Database Setup

The application automatically creates necessary tables on first run:
- Users and authentication
- Expenses, income, and budgets
- Subscription plans and user subscriptions

## Configuration

Create a `.streamlit/config.toml` file:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

## Usage

1. Register a new account or login
2. Start by setting your monthly income
3. Add expenses and categorize them
4. Set budgets for different categories
5. Upgrade to premium plans for advanced features

## Premium Features Access

- Charts & Analytics: Available with Basic and Premium plans
- Future Predictions: Premium plan only
- Unlimited expenses: Premium plan only
