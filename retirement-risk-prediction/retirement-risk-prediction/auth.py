import streamlit as st
import json
import os

USERS_FILE = "users.json"

def verify_password(password, stored_password):
    return password == stored_password

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def init_default_users():
    users = load_users()
    if not users:
        users = {
            "admin": {
                "password": "admin123",
                "name": "관리자"
            },
            "user": {
                "password": "user123", 
                "name": "사용자"
            }
        }
        save_users(users)
    return users

def login(username, password):
    users = load_users()
    if username in users and verify_password(password, users[username]["password"]):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.user_name = users[username]["name"]
        return True
    return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_name = None

def is_logged_in():
    return st.session_state.get("logged_in", False)

def get_current_user():
    return {
        "username": st.session_state.get("username"),
        "name": st.session_state.get("user_name")
    }

def show_login_page():
    st.title("🔐 퇴직연금 리스크 예측 시스템 로그인")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                st.markdown("### 로그인")
                username = st.text_input("사용자명")
                password = st.text_input("비밀번호", type="password")
                submit = st.form_submit_button("로그인", use_container_width=True)
                
                if submit:
                    if login(username, password):
                        st.success("로그인 성공!")
                        st.rerun()
                    else:
                        st.error("사용자명 또는 비밀번호가 잘못되었습니다.")

def show_logout_button():
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("로그아웃"):
            logout()
            st.rerun()
    with col1:
        user = get_current_user()
        st.markdown(f"**{user['name']}**님, 안녕하세요!")