# 🧾 Policy Optimization for Financial Decision-Making  
## Task 4: Analysis, Comparison, and Future Steps  

---

## 1️⃣ Presenting the Results  

### 🔹 Supervised Deep Learning Model  
The **MLP classifier** was trained to predict whether a borrower would default or fully pay the loan.  

**Performance Metrics:**

| Metric | Value | Interpretation |
|---------|--------|----------------|
| **AUC** | **0.788** | The model can correctly rank defaulters vs. non-defaulters 78.8 % of the time. |
| **F1-Score** | **0.407** | Moderate balance between precision and recall; limited by class imbalance. |
| **Accuracy** | **0.659** | 65.9 % of overall predictions are correct. |

**Confusion Matrix**  
`[[146578, 83618], [8605, 31674]]`  

→ The model tends to over-predict defaults (many false positives) but captures most real defaulters.  

---

### 🔹 Offline Reinforcement Learning Agent  
The **CQL (Conservative Q-Learning)** agent was trained to maximize expected financial return using historical loan data.  

**Reward Function:**
- Approve + Paid → `+ (loan_amnt × int_rate)`  
- Approve + Default → `– loan_amnt`  
- Deny → `0`  

**Evaluation Metric:**
- **Estimated Policy Value (EPV)** — average realized reward per decision under learned policy.  

**Observed Result:**  
The CQL policy consistently achieved a **positive EPV**, meaning its approval strategy yields a **net profit** across the test set, unlike random or threshold-based baselines.

---

## 2️⃣ Explaining the Metrics  

### 🧠 Why AUC & F1 for the DL Model?  
- **AUC (Area Under ROC Curve)** evaluates **ranking ability** — crucial when business teams must set risk thresholds.  
- **F1-Score** balances **false positives (unnecessary rejections)** and **false negatives (missed risky loans)**.  
Together, they measure how well the model discriminates borrowers and manages credit risk — the key purpose of a **risk assessment model**.

### 💰 Why Estimated Policy Value for the RL Agent?  
- The RL objective is not classification accuracy, but **maximizing expected profit**.  
- **EPV** directly measures the **average monetary outcome** if the policy were deployed in production.  
- It captures the trade-off between risk and reward — a policy with a few defaults can still be optimal if it earns more interest overall.  

---

## 3️⃣ Policy Comparison  

| Aspect | Supervised DL Model | Offline RL Agent |
|--------|----------------------|------------------|
| Objective | Predict default probability | Maximize long-term financial reward |
| Training Signal | Binary cross-entropy (label = default) | Reward function (profit or loss) |
| Decision Rule | Approve if p(default) < threshold | Approve if expected_reward > 0 |
| Typical Behavior | Conservative — avoids defaults | Opportunistic — accepts profitable risk |
| Evaluation Metric | AUC / F1 | Estimated Policy Value (EPV) |

**Example Scenario:**  
- Applicant A: High income, moderate debt, interest rate 18 %.  
  - **DL model:** predicts p(default)=0.55 → Rejects.  
  - **RL agent:** calculates expected_reward > 0 → Approves.  
  - **Reason:** although default probability > 50 %, the **interest income** outweighs the **expected loss**.  

This difference illustrates the core shift:  
- The **DL model** minimizes *error*.  
- The **RL agent** maximizes *profit*.  

---

## 4️⃣ Future Steps & Recommendations  

### 🚀 Deployment Strategy  
- **Short term:** Deploy the **DL model** for risk scoring (stable, explainable).  
- **Mid term:** Integrate RL policy in **simulation or shadow mode** to validate profitability without affecting real loans.  
- **Long term:** Use a **hybrid approach** — DL risk estimator + RL policy optimizer.  

### ⚠️ Limitations  
1. **Dataset Bias:** Historical data only includes *approved* loans (offline RL constraint).  
2. **Simplified Reward Function:** Ignores partial repayments, recoveries, and time value of money.  
3. **Data Imbalance:** Fewer default cases lower F1 and bias supervised training.  
4. **Explainability:** RL decisions are less transparent; requires interpretability tools (e.g., SHAP for RL).  

### 🧩 Future Work  
- Collect **rejected-loan data** for unbiased counterfactual training.  
- Incorporate **time-series repayment behavior** into state representation.  
- Explore **distributional RL** and **risk-sensitive objectives** (CVaR, quantile Q-learning).  
- Compare CQL with **BCQ, TD3+BC, IQL** for robustness.  
- Integrate **causal inference** to estimate impact of approval policy on market default rates.  

---

## 5️⃣ Conclusion  
The supervised model provides a reliable **credit-risk assessment**, while the offline RL agent offers a **profit-maximizing decision policy**.  
Both are valuable:  
- The **DL model** ensures prudence.  
- The **RL agent** ensures profitability.  

Combining them forms a practical, data-driven, and financially optimized loan-approval system.
