# Policy Optimization for Financial Decision-Making

This project demonstrates an **end-to-end pipeline** combining **Supervised Deep Learning** and **Offline Reinforcement Learning (RL)** to optimize **loan approval decisions** in a real-world financial context.

---

## 📈 Objective

The goal is to design an intelligent system that decides **whether to approve or deny a new loan** to **maximize profit while minimizing risk**.  
This project mirrors the work of a **FinTech Research Scientist** using ML + RL for credit decision-making.

**Skills demonstrated:**
- Data analysis & feature engineering  
- Building and evaluating deep learning models  
- Framing supervised learning as an **offline RL** problem  
- Applying CQL (Conservative Q-Learning) for policy optimization  
- Interpreting, comparing, and communicating model behaviors  

---

## 🧩 Dataset — LendingClub Loan Data

**Source:** [Kaggle – Lending Club Loan Data (2007–2018)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

- Contains borrower info, loan details, and repayment outcomes.  
- Focus file: `accepted_2007_to_2018Q4.csv`  
- Target variable: **loan_status**  
  - `0 → Fully Paid`  
  - `1 → Defaulted / Charged Off`

> ⚠️ The full dataset is large (1GB+).  
> Download it manually from Kaggle and place it in `/data/`.

