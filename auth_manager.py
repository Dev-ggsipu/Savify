import streamlit as st
import hashlib
import secrets
from datetime import datetime, timedelta
from db_manager import DatabaseManager
import json

class AuthManager:
    def __init__(self):
        self.db_manager = DatabaseManager()
        # Initialize subscription plans
        self.db_manager.initialize_subscription_plans()
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return salt + pwd_hash.hex()
    
    def verify_password(self, stored_password: str, provided_password: str) -> bool:
        """Verify password against stored hash"""
        salt = stored_password[:32]
        stored_hash = stored_password[32:]
        pwd_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return pwd_hash.hex() == stored_hash
    
    def generate_user_id(self, email: str) -> str:
        """Generate unique user ID from email"""
        return hashlib.md5(email.encode()).hexdigest()
    
    def register_user(self, email: str, name: str, password: str) -> dict:
        """Register a new user"""
        user_id = self.generate_user_id(email)
        
        # Check if user already exists
        existing_user = self.db_manager.get_user(user_id)
        if existing_user:
            return {'success': False, 'message': 'User already exists with this email'}
        
        # Create user
        success = self.db_manager.create_user(user_id, email, name)
        if success:
            return {'success': True, 'message': 'Account created successfully', 'user_id': user_id}
        else:
            return {'success': False, 'message': 'Failed to create account'}
    
    def login_user(self, email: str) -> dict:
        """Login user (simplified for demo)"""
        user_id = self.generate_user_id(email)
        user = self.db_manager.get_user(user_id)
        
        if user:
            return {'success': True, 'user': user}
        else:
            # Auto-register for demo purposes
            name = email.split('@')[0].title()
            result = self.register_user(email, name, "demo_password")
            if result['success']:
                user = self.db_manager.get_user(user_id)
                return {'success': True, 'user': user}
            return {'success': False, 'message': 'User not found'}
    
    def get_subscription_plans(self) -> list:
        """Get formatted subscription plans"""
        plans_df = self.db_manager.get_subscription_plans()
        if plans_df.empty:
            return []
        
        plans = []
        for _, plan in plans_df.iterrows():
            try:
                features = json.loads(plan['features']) if plan['features'] else []
            except (json.JSONDecodeError, TypeError):
                features = []
            plans.append({
                'name': plan['plan_name'],
                'display_name': plan['display_name'],
                'price': plan['price'],
                'billing_cycle': plan['billing_cycle'],
                'description': plan['description'],
                'features': features,
                'max_expenses': plan['max_expenses'],
                'max_categories': plan['max_categories'],
                'has_predictions': plan['has_predictions'] == 'true',
                'has_analytics': plan['has_analytics'] == 'true',
                'has_budgets': plan['has_budgets'] == 'true',
                'has_export': plan['has_export'] == 'true'
            })
        
        return plans
    
    def upgrade_subscription(self, user_id: str, plan_name: str) -> dict:
        """Upgrade user subscription (simplified for demo)"""
        # In production, this would integrate with payment processor
        start_date = datetime.now()
        end_date = start_date + timedelta(days=30)  # Monthly subscription
        
        success = self.db_manager.update_user_subscription(
            user_id, plan_name, start_date, end_date, f"demo_payment_{user_id}"
        )
        
        if success:
            return {'success': True, 'message': f'Successfully upgraded to {plan_name} plan'}
        else:
            return {'success': False, 'message': 'Failed to upgrade subscription'}
    
    def check_feature_access(self, user_id: str, feature: str) -> bool:
        """Check if user has access to specific feature"""
        limits = self.db_manager.check_user_limits(user_id)
        if not limits.get('valid', False):
            return False
        
        feature_map = {
            'predictions': limits.get('has_predictions', False),
            'analytics': limits.get('has_analytics', False),
            'budgets': limits.get('has_budgets', False),
            'export': limits.get('has_export', False)
        }
        
        return feature_map.get(feature, False)
    
    def show_login_page(self):
        """Display login/registration page"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #2E86AB; margin-bottom: 0.5rem;">💰 Savify = just save it !!!</h1>
            <p style="color: #6c757d; font-size: 1.2rem;">Smart Financial Management</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            st.subheader("Welcome Back!")
            
            with st.form("login_form"):
                email = st.text_input("Email Address", placeholder="your.email@example.com")
                submitted = st.form_submit_button("Login", use_container_width=True)
                
                if submitted and email:
                    result = self.login_user(email)
                    if result['success']:
                        st.session_state.user = result['user']
                        st.session_state.authenticated = True
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result['message'])
                elif submitted:
                    st.error("Please enter your email address")
        
        with tab2:
            st.subheader("Create Your Account")
            
            with st.form("register_form"):
                name = st.text_input("Full Name", placeholder="John Doe")
                email = st.text_input("Email Address", placeholder="your.email@example.com")
                submitted = st.form_submit_button("Create Account", use_container_width=True)
                
                if submitted and name and email:
                    result = self.register_user(email, name, "demo_password")
                    if result['success']:
                        # Auto-login after registration
                        login_result = self.login_user(email)
                        if login_result['success']:
                            st.session_state.user = login_result['user']
                            st.session_state.authenticated = True
                            st.success("Account created and logged in successfully!")
                            st.rerun()
                    else:
                        st.error(result['message'])
                elif submitted:
                    st.error("Please fill in all fields")
    
    def show_subscription_page(self):
        """Display subscription plans and upgrade options"""
        st.markdown('<div class="section-header"><h2>💎 Subscription Plans</h2></div>', unsafe_allow_html=True)
        
        user = st.session_state.get('user', {})
        current_plan = user.get('subscription_plan', 'free')
        
        # Current plan info
        st.subheader("Current Plan")
        limits = self.db_manager.check_user_limits(user['user_id'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Plan", limits.get('plan_name', 'Free Plan'))
        with col2:
            expenses_used = limits.get('expenses_used', 0)
            expenses_limit = limits.get('expenses_limit', 50)
            limit_text = str(expenses_limit) if expenses_limit != -1 else "Unlimited"
            st.metric("Expenses This Month", f"{expenses_used}/{limit_text}")
        with col3:
            categories_used = limits.get('categories_used', 0)
            categories_limit = limits.get('categories_limit', 5)
            limit_text = str(categories_limit) if categories_limit != -1 else "Unlimited"
            st.metric("Categories", f"{categories_used}/{limit_text}")
        
        # Available plans
        st.subheader("Available Plans")
        plans = self.get_subscription_plans()
        
        cols = st.columns(len(plans))
        
        for i, plan in enumerate(plans):
            with cols[i]:
                # Plan card styling
                is_current = plan['name'] == current_plan
                border_color = "#2E86AB" if is_current else "#e9ecef"
                background = "#f8f9fa" if is_current else "#ffffff"
                
                # Plan card using native Streamlit components
                border_style = "border: 2px solid #2E86AB; background: #f8f9fa;" if is_current else "border: 1px solid #e9ecef; background: white;"
                
                with st.container():
                    st.markdown(f"""
                    <div style="{border_style} border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
                        <h3 style="color: #2E86AB; margin-bottom: 0.5rem;">{plan['display_name']}</h3>
                        <div style="font-size: 2rem; font-weight: bold; color: #333; margin: 1rem 0;">
                            {'Free' if plan['price'] == 0 else f"${plan['price']:.2f}"}
                        </div>
                        <div style="color: #6c757d; font-size: 0.9rem; margin-bottom: 1rem;">per month</div>
                        <p style="color: #6c757d; margin-bottom: 1.5rem;">{plan['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Features:**")
                    for feature in plan['features']:
                        st.markdown(f"✅ {feature}")
                    
                    st.markdown("")  # Add some spacing
                
                if is_current:
                    st.markdown("""
                    <div style="text-align: center; margin-top: 1rem;">
                        <span style="background: #2E86AB; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                            Current Plan
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                elif plan['name'] != 'free':
                    if st.button(f"Upgrade to {plan['display_name']}", key=f"upgrade_{plan['name']}", use_container_width=True):
                        result = self.upgrade_subscription(user['user_id'], plan['name'])
                        if result['success']:
                            st.success(result['message'])
                            # Update session user data
                            updated_user = self.db_manager.get_user(user['user_id'])
                            st.session_state.user = updated_user
                            st.rerun()
                        else:
                            st.error(result['message'])
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature comparison
        st.subheader("Feature Comparison")
        
        comparison_data = []
        features_list = ["Basic Expense Tracking", "Budget Management", "Advanced Analytics", 
                        "AI Predictions", "Data Export", "Premium Support"]
        
        for feature in features_list:
            row = {"Feature": feature}
            for plan in plans:
                if "Basic expense tracking" in feature.lower() or "budget" in feature.lower():
                    row[plan['display_name']] = "✓" if plan['has_budgets'] else "✗"
                elif "analytics" in feature.lower():
                    row[plan['display_name']] = "✓" if plan['has_analytics'] else "✗"
                elif "predictions" in feature.lower():
                    row[plan['display_name']] = "✓" if plan['has_predictions'] else "✗"
                elif "export" in feature.lower():
                    row[plan['display_name']] = "✓" if plan['has_export'] else "✗"
                elif "support" in feature.lower():
                    row[plan['display_name']] = "✓" if plan['name'] != 'free' else "✗"
                else:
                    row[plan['display_name']] = "✓"
            
            comparison_data.append(row)
        
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)