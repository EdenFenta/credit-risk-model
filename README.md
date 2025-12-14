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

### Why We Need a Proxy Default Variable

In credit risk modeling, we often do not observe loan defaults directly due to limited data or observational constraints. This necessitates the use of a proxy default variable:

- Behavioral data is used as a substitute to infer default likelihood.
- RFM (Recency, Frequency, Monetary value) metrics capture customer engagement and payment likelihood.
- Using a proxy introduces noise and business risk, as it may not perfectly align with actual defaults, potentially leading to less accurate predictions or increased uncertainty.

### Trade-offs: Simple vs Complex Models

Accuracy is not the only goal in finance; considerations like interpretability, regulatory compliance, and validation ease are crucial. Below is a comparison of simple and complex models:

| Aspect                | Simple Models          | Complex Models          |
|-----------------------|------------------------|-------------------------|
| Interpretability      | Interpretable          | Hard to explain         |
| Regulatory Compliance | Regulatory-friendly    | Higher compliance risk  |
| Validation            | Easier validation      | Higher accuracy         |
