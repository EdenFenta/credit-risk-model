# Credit Risk Model for Buy Now Pay Later

## Credit Scoring Business Understanding

### Influence of Basel II on Interpretable Models

Basel II is a regulatory framework that emphasizes risk management in banking. It requires banks to:

- Quantify risk
- Justify decisions
- Document assumptions

This influences the need for interpretable models in the following ways:

- Models must be explainable to ensure transparency in risk assessment.
- Regulators may audit decisions, requiring clear rationale behind model outputs.
- Black-box models increase compliance risk, as they can be difficult to justify or audit effectively.

#### Basel II Regulatory Context and Supervisory Expectations

Under the Basel II Capital Accord, credit risk modeling is governed primarily by Pillar 1 (Minimum Capital Requirements) and Pillar 2 (Supervisory Review Process).

Pillar 1 requires banks to quantify credit risk in a consistent and conservative manner to determine capital adequacy. This places emphasis on reliable risk estimates, stable model performance, and defensible assumptions.

Pillar 2 empowers regulators to challenge a bank’s internal risk models if they are insufficiently transparent, poorly documented, or inadequately validated. For example, a regulator may require the bank to explain why two customers with similar observable behavior received materially different credit decisions. If the model relies on a complex black-box algorithm without clear feature attribution or justification, the bank may be required to apply higher capital buffers or restrict model usage.

As a result, model interpretability, documentation, and governance are not optional design choices but regulatory necessities.

### Why We Need a Proxy Default Variable

In credit risk modeling, we often do not observe loan defaults directly due to limited data or observational constraints. This necessitates the use of a proxy default variable:

- Behavioral data is used as a substitute to infer default likelihood.
- RFM (Recency, Frequency, Monetary value) metrics capture customer engagement and payment likelihood.
- Using a proxy introduces noise and business risk, as it may not perfectly align with actual defaults, potentially leading to less accurate predictions or increased uncertainty.

#### Alternative Proxy Definitions and Business Risk Implications

In the absence of an observed default label, this project constructs a proxy target variable based on customer engagement patterns derived from RFM analysis. While disengagement is a reasonable indicator of elevated credit risk, alternative proxy definitions could be considered, each with distinct trade-offs.

One alternative proxy is transaction volatility, where customers exhibiting highly irregular spending patterns are labeled as higher risk. This may better capture behavioral instability but could bias the model against seasonal or irregular-income customers.

Another proxy could be repeated refund or reversal behavior, which may indicate financial stress or misuse. While more directly tied to repayment risk, such signals may be sparse and introduce class imbalance.

A third alternative is channel-based risk segmentation, where certain channels (e.g., pay-later heavy usage without follow-up transactions) signal higher risk. This may improve short-term performance but risks embedding operational or platform-specific bias.

The chosen RFM-based proxy balances data availability, interpretability, and regulatory defensibility, but its limitations must be acknowledged. Predictions derived from proxy labels should therefore be used probabilistically and monitored continuously for bias and performance drift.

### Trade-offs: Simple vs Complex Models

Accuracy is not the only goal in finance; considerations like interpretability, regulatory compliance, and validation ease are crucial. Below is a comparison of simple and complex models:

| Aspect                | Simple Models          | Complex Models          |
|-----------------------|------------------------|-------------------------|
| Interpretability      | Interpretable          | Hard to explain         |
| Regulatory Compliance | Regulatory-friendly    | Higher compliance risk  |
| Validation            | Easier validation      | Higher accuracy         |

## Exploratory Data Analysis (EDA)

### Key Insights

1. Transaction amounts are highly skewed, requiring scaling or transformation.
2. A small number of customers account for a large proportion of transactions.
3. Certain product categories dominate transaction volume.
4. Numerical features show limited linear correlation.
