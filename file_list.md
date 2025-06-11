# Savify Application Files

## Core Application Files (Required)

### 1. app.py
**Main application file** - Contains the Streamlit interface, navigation, and all page functions
- Size: ~35KB
- Contains: UI components, dashboard, expense management, analytics, predictions

### 2. auth_manager.py  
**Authentication & Subscription System** - Handles user login, registration, and subscription management
- Size: ~15KB
- Contains: User authentication, subscription plans, upgrade interface

### 3. db_manager.py
**Database Operations** - PostgreSQL database models and operations
- Size: ~25KB  
- Contains: SQLAlchemy models, database operations, user limits checking

### 4. data_manager.py
**Legacy Data Operations** - CSV-based data management (backup system)
- Size: ~12KB
- Contains: CSV file operations for expenses, income, budgets

### 5. pyproject.toml
**Project Dependencies** - Python package requirements
- Size: 1KB
- Contains: All required Python packages and versions

## Configuration Files

### 6. README.md
**Documentation** - Complete setup and usage instructions
- Size: 3KB
- Contains: Installation guide, features overview, usage instructions

## Optional Files (Generated during usage)

### 7. .streamlit/config.toml (Create this file)
**Streamlit Configuration**
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

## Environment Setup

### Required Environment Variables:
- `DATABASE_URL` - PostgreSQL connection string

### Required Python Packages:
- streamlit
- pandas  
- plotly
- scikit-learn
- sqlalchemy
- psycopg2-binary
- numpy

## Download Instructions

1. Copy all core application files (1-5) to your project directory
2. Create the `.streamlit/config.toml` file with the configuration above  
3. Install dependencies: `pip install streamlit pandas plotly scikit-learn sqlalchemy psycopg2-binary numpy`
4. Set up PostgreSQL database and configure DATABASE_URL
5. Run: `streamlit run app.py --server.port 5000`

## File Descriptions

- **app.py**: Complete Streamlit application with all features
- **auth_manager.py**: User authentication and subscription system
- **db_manager.py**: Database models and operations for PostgreSQL
- **data_manager.py**: Legacy CSV operations (fallback system)
- **pyproject.toml**: Python dependencies configuration
- **README.md**: Complete documentation and setup guide