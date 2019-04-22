# Estimated Delivery Date (EDD) - Case study

## 1 Case study

#### Case study questions

Given the context below, you are expected to answer the following:

1. Build a model that estimates accurately the Expected Delivery Time
2. Analyze patterns, anomalies that can be used to explain variations to the Expected Delivery Time
3. Given additional instances of data (>20M rows) which challenges do you expect to face? How can you overcome them?
4. Given access to additional data, which features could be useful to include in your model?
5. How would you go about operationalizing your solution in a production environment? Can you devise a possible architecture that can answer around 5 prediction requests per second to your model? (and growing)

Please summarize your findings in a small presentation in English that we will ask you to present during the interview phase (30 minutes). 

Share all the work that you developed to support your findings (i.e. Jupyter notebooks, R workflow, SQL, Excel).

When you're done, send the results compressed directly to the recruiter.

#### Deliverables

We expect a compressed file containing:
1. The results, containing the `OrderLineID` and the respective `DeliveryTime`
2. A short presentation with your solution, including main findings, methodology and prediction strategy
3. All supporting materials (e.g. Jupyter notebooks, Python package, R workflow, SQL, Excel)

### 1.1 About Fraud 

We are thrilled to share with you this challenge for the interview process for the **Data Science Engineer role in the Fraud team of Farfetch**. This stage is essential to assess your ability to structure and develop an intelligent solution to a real business problem, that represents a critical step in Farfetch: the order creation and fulfillment.

Farfetch built its platform and business centered around its customers. The Fraud team at Farfetch works everyday to give customers a safe and enhanced shopping journey, while minimizing the fraud-related losses for the company and its partners. Our task is to block the creation and fulfillment of fraudulent orders, while allowing all the legitimate ones.


### 1.2 About the Estimated Delivery Date (EDD)

When shopping at Farfetch, customers are offered a catalog of thousands of products that are sourced from different partners spread around the world. During the shopping experience, customers must have an accurate estimate of their purchase delivery date to manage their expectations.

The goal of this case study is to **understand the Expected Delivery Date of an order and provide a robust and accurate delivery date estimate** to the customer. To support your understanding of the problem and development of the challenge you will receive a dataset split in training and test set. Further details are given in the Data Instructions attached to the case.

### 1.3 About the case study questions

You can contact us for any questions regarding the case or the data model.

## 2 About the project structure

Here's how the present project is organized:

```
.
├── data
│   ├── test.csv
│   └── train.csv
└── README.md
```

## 3 About the data

The data made available for this challenge is described in the instructions below. Note that the dataset is sampled, anonymized and might not represent reality. Even so, do not publish, share or make this data available to outside parties (i.e. public code repositories). Failing to oblige will impact your application.

| Variable            | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `OrderLineID`       | Identifier of the item                                                      |
| `OrderCodeID`       | Identifier of the order                                                     |
| `PartnerID`         | Identifier of the partner                                                   |
| `CustomerCountry`   |	Country the order is shipping to                                            |
| `CustomerCity`	    | City the order is shipping to                                               |
| `CustomerLatitude`  |	Customer’s geographic coordinates (latitude)                                |
| `CustomerLongitude`	| Customer’s geographic coordinates (longitude)                               |
| `PartnerCountry`    |	Country the order is shipping from                                          |
| `PartnerCity`	      | City the order is shipping from                                             |
| `PartnerLatitude`   |	Partner’s geographic coordinates (latitude)                                 |
| `PartnerLongitude`  |	Partner’s geographic coordinates (longitude)                                |
| `DeliveryType`      |	Identifies the delivery type chosen by the user                             |
| `IsHazmat`          |	Contains hazardous materials (i.e. batteries, beauty items)                 |
| `TariffCode`	      | Identifier of the applied tariff                                            |
| `DDPCategory`       | Commodity category classification (for [HS Code](https://goo.gl/PtwHap))    |
| `DDPSubCategory`	  | Commodity subcategory classification (for [HS Code](https://goo.gl/PtwHap)) |
| `Category1stLevel`  | High-level category of the product on the website                           |
| `Category2ndLevel`  | Lower-level category of the product on the website                          |
| `OrderDate`         | Order date and time                                                         |
| `DeliveryTime`	    | Delivery time, in days                                      |
