import streamlit as st
import requests
from datetime import datetime

API_BASE = "https://railpartner-ai.onrender.com"

st.set_page_config(
    page_title="RailPartner AI",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal safe CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
div[data-testid="stForm"] { border: none; padding: 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None

# ── API helpers ────────────────────────────────────────────────────────────────
def api_post(endpoint, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, headers=headers, timeout=30)
        return r.status_code, r.json() if r.content else {}
    except requests.exceptions.ConnectionError:
        return 503, {"detail": "Server is waking up, try again in 30 seconds."}
    except Exception as e:
        return 500, {"detail": str(e)}

def api_post_form(endpoint, data):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", data=data, timeout=30)
        return r.status_code, r.json() if r.content else {}
    except requests.exceptions.ConnectionError:
        return 503, {"detail": "Server is waking up, try again in 30 seconds."}
    except Exception as e:
        return 500, {"detail": str(e)}

def api_get(endpoint, token, params=None):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(f"{API_BASE}{endpoint}", headers=headers, params=params, timeout=30)
        return r.status_code, r.json() if r.content else {}
    except requests.exceptions.ConnectionError:
        return 503, {"detail": "Server is waking up, try again in 30 seconds."}
    except Exception as e:
        return 500, {"detail": str(e)}

def api_delete(endpoint, token):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.delete(f"{API_BASE}{endpoint}", headers=headers, timeout=30)
        return r.status_code, {}
    except Exception as e:
        return 500, {"detail": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.token:

    st.title("🚂 RailPartner AI")
    st.caption("Predict crowd levels and seat availability before you board.")
    st.divider()

    tab_login, tab_reg = st.tabs(["Sign in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in →", use_container_width=True)

        if submitted:
            if username and password:
                with st.spinner("Signing in…"):
                    code, data = api_post_form(
                        "/api/auth/login",
                        {"username": username, "password": password}
                    )
                if code == 200:
                    st.session_state.token = data["access_token"]
                    st.session_state.user  = data["user"]
                    st.rerun()
                else:
                    st.error(data.get("detail", "Login failed"))
            else:
                st.warning("Please fill in both fields.")

    with tab_reg:
        with st.form("register_form"):
            r_email = st.text_input("Email")
            r_user  = st.text_input("Username")
            r_pass  = st.text_input("Password (min 8 characters)", type="password")
            submitted_r = st.form_submit_button("Create account →", use_container_width=True)

        if submitted_r:
            if r_email and r_user and r_pass:
                with st.spinner("Creating account…"):
                    code, data = api_post(
                        "/api/auth/register",
                        {"email": r_email, "username": r_user, "password": r_pass}
                    )
                if code == 201:
                    st.success("✅ Account created! Switch to Sign in tab.")
                else:
                    st.error(data.get("detail", "Registration failed"))
            else:
                st.warning("Please fill in all fields.")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# Top bar
col_title, col_user = st.columns([4, 1])
with col_title:
    st.title("🚂 RailPartner AI")
with col_user:
    uname = st.session_state.user.get("username", "User") if st.session_state.user else "User"
    st.markdown(f"<br>👤 **{uname}**", unsafe_allow_html=True)
    if st.button("Sign out"):
        st.session_state.token = None
        st.session_state.user  = None
        st.rerun()

st.divider()

tab_pred, tab_hist = st.tabs(["🔍 Predict", "📋 History"])


# ── PREDICT TAB ───────────────────────────────────────────────────────────────
with tab_pred:

    with st.form("predict_form"):
        st.subheader("Journey details")
        c1, c2, c3 = st.columns(3)
        with c1:
            journey_type = st.selectbox(
                "Train type",
                ["express", "superfast", "local", "mail", "intercity", "rajdhani"],
                index=1
            )
        with c2:
            distance = st.number_input("Distance (km)", min_value=1, max_value=5000, value=450, step=10)
        with c3:
            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=lambda m: datetime(2024, m, 1).strftime("%B"),
                index=5
            )

        c4, c5 = st.columns(2)
        with c4:
            is_weekend = st.checkbox("Weekend journey", value=True)
        with c5:
            holiday_type = st.selectbox(
                "Holiday type",
                [0, 1, 2],
                format_func=lambda x: ["None", "Public holiday", "Festival season"][x]
            )

        st.subheader("Coach capacity")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1: sl_capacity  = st.number_input("SL capacity",  min_value=0, value=300, step=10)
        with cc2: ac3_capacity = st.number_input("3A capacity",  min_value=0, value=150, step=10)
        with cc3: ac2_capacity = st.number_input("2A capacity",  min_value=0, value=100, step=10)
        with cc4: ac1_capacity = st.number_input("1A capacity",  min_value=0, value=50,  step=10)

        st.subheader("Current bookings")
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1: sl_booked  = st.number_input("SL booked",  min_value=0, value=240, step=10)
        with bc2: ac3_booked = st.number_input("3A booked",  min_value=0, value=120, step=10)
        with bc3: ac2_booked = st.number_input("2A booked",  min_value=0, value=50,  step=10)
        with bc4: ac1_booked = st.number_input("1A booked",  min_value=0, value=20,  step=10)

        st.write("")
        submitted = st.form_submit_button("🔍 Predict now", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "is_weekend":   is_weekend,
            "holiday_type": holiday_type,
            "distance":     float(distance),
            "month":        month,
            "journey_type": journey_type,
            "sl_capacity":  sl_capacity,
            "ac3_capacity": ac3_capacity,
            "sl_booked":    sl_booked,
            "ac3_booked":   ac3_booked,
            "ac2_booked":   ac2_booked,
            "ac1_booked":   ac1_booked,
            "ac2_capacity": ac2_capacity,
            "ac1_capacity": ac1_capacity,
        }
        with st.spinner("Running prediction…"):
            code, data = api_post("/api/predict", payload, st.session_state.token)

        if code == 200:
            crowd = data["crowd"]
            seat  = data["seat"]

            st.divider()
            st.subheader("Prediction results")

            r1, r2 = st.columns(2)

            with r1:
                st.markdown("#### 👥 Crowd Level")
                crowd_label = crowd["label"]
                conf        = crowd["confidence"]

                if crowd_label == "High":
                    st.error(f"**{crowd_label}** — {conf:.1%} confidence")
                elif crowd_label == "Medium":
                    st.warning(f"**{crowd_label}** — {conf:.1%} confidence")
                else:
                    st.success(f"**{crowd_label}** — {conf:.1%} confidence")

                st.progress(conf)
                st.caption("Probability breakdown")
                for k, v in sorted(crowd["probabilities"].items(), key=lambda x: -x[1]):
                    st.markdown(f"- **{k}**: {v:.1%}")

            with r2:
                st.markdown("#### 💺 Seat Availability")
                seat_label = seat["label"]
                conf2      = seat["confidence"]

                if "High" in seat_label:
                    st.success(f"**{seat_label}** — {conf2:.1%} confidence")
                elif "Medium" in seat_label:
                    st.warning(f"**{seat_label}** — {conf2:.1%} confidence")
                else:
                    st.error(f"**{seat_label}** — {conf2:.1%} confidence")

                st.progress(conf2)
                st.caption("Probability breakdown")
                for k, v in sorted(seat["probabilities"].items(), key=lambda x: -x[1]):
                    st.markdown(f"- **{k}**: {v:.1%}")

        elif code == 401:
            st.error("Session expired. Please sign in again.")
            st.session_state.token = None
            st.session_state.user  = None
            st.rerun()
        else:
            st.error(f"Prediction failed: {data.get('detail', 'Unknown error')}")


# ── HISTORY TAB ───────────────────────────────────────────────────────────────
with tab_hist:

    col_a, col_b = st.columns([3, 1])
    with col_b:
        limit = st.selectbox("Show last", [10, 20, 50], index=0)
    with col_a:
        st.write("")
        refresh = st.button("🔄 Refresh", key="refresh_hist")

    code, data = api_get(
        "/api/predict/history",
        st.session_state.token,
        params={"limit": limit}
    )

    if code == 200:
        if not data:
            st.info("No predictions yet. Run your first prediction in the Predict tab.")
        else:
            st.caption(f"Showing {len(data)} most recent predictions")
            st.divider()

            for rec in data:
                dt     = datetime.fromisoformat(rec["created_at"].replace("Z", "+00:00"))
                dt_str = dt.strftime("%d %b %Y · %H:%M")
                jtype  = (rec.get("journey_type") or "Unknown").capitalize()

                col_info, col_crowd, col_seat, col_del = st.columns([3, 2, 2, 1])

                with col_info:
                    st.markdown(f"**{jtype}** · {rec['distance']:.0f} km")
                    st.caption(dt_str)

                with col_crowd:
                    cl = rec["crowd_level"]
                    if cl == "High":
                        st.error(f"👥 {cl}")
                    elif cl == "Medium":
                        st.warning(f"👥 {cl}")
                    else:
                        st.success(f"👥 {cl}")

                with col_seat:
                    ss = rec["seat_status"]
                    if "High" in ss:
                        st.success(f"💺 {ss}")
                    elif "Medium" in ss:
                        st.warning(f"💺 {ss}")
                    else:
                        st.error(f"💺 {ss}")

                with col_del:
                    if st.button("🗑️", key=f"del_{rec['id']}", help="Delete"):
                        d_code, _ = api_delete(
                            f"/api/predict/history/{rec['id']}",
                            st.session_state.token
                        )
                        if d_code == 204:
                            st.success("Deleted")
                            st.rerun()
                        else:
                            st.error("Failed")

                st.divider()

    elif code == 401:
        st.error("Session expired. Please sign in again.")
        st.session_state.token = None
        st.session_state.user  = None
        st.rerun()
    else:
        st.error(f"Could not load history: {data.get('detail', 'Unknown error')}")