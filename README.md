# 🌱 Tree Survival Planner  
**AI-powered Decision Support System for Sustainable Tree Plantation**

---

## 📌 Overview

**Tree Survival Planner** is an interactive, AI-driven web application designed to help planners, foresters, and researchers **reduce plantation failure** by estimating **tree survival probability over time** and identifying **optimal planting conditions**.

Instead of answering *“Will the tree survive?”*, this system answers the more meaningful question:

> **“How long is a tree likely to survive under given conditions?”**

The application uses **survival analysis (Weibull AFT model)** combined with an intuitive decision-oriented UI to support **data-informed plantation planning**.

---

## 🎯 Key Objectives

- Estimate **36-month survival probability** of trees  
- Identify **risk levels** (Low / Medium / High)  
- Recommend **best planting windows**  
- Allow **what-if analysis** using interactive inputs  
- Provide **explainable insights**, not black-box predictions  

---

## 🧠 Why Survival Analysis?

Traditional machine learning models treat survival as a binary outcome (alive/dead).  
This project uses **survival analysis**, which:

- Models **time-to-event** explicitly  
- Handles **censored data** (trees still alive at last observation)  
- Produces **survival curves**, not just point predictions  
- Is widely used in **medical, ecological, and reliability studies**

This makes the system **more realistic and defensible** for long-term planning.

---

## 🏗️ System Architecture

**High-level components:**

- **Frontend:** Streamlit (mobile-friendly web UI)
- **Backend Logic:** Python-based preprocessing & decision logic
- **ML Model:** Weibull Accelerated Failure Time (AFT) model
- **Artifacts:** Serialized model & encoders loaded via `joblib`

**Flow:**
User Input → Feature Processing → Survival Model → Survival Probability → Risk Classification → Visual Insights

