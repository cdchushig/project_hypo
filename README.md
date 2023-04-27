# Severe hypoglucemia

In this project, we have used and analyzed clinical data from the Jaeb Center for Health Research, Tampa, Florida (USA), in particular, data collected by the T1D Exchange study (Ruth S Weinstock et al. Risk factors associated with severe hypoglycemia in older adults with type 1 diabetes. Diabetes Care, 39(4):603–610, 2016).

This database consists of 16 surveys (taken from https://t1dexchange.org/), which contain information regarding medications, medical conditions, demographic information, alcohol consumption, nutrition, insulin use, blood tests, and tests to assess their cognition or attitude towards certain situations. 
The available data corresponds to 201 patients, divided into 100 control and 101 case patients. On the one hand, the case patient is identified as one who has suffered a severe hypoglycemia event in the last 12 months, defined as an event that requires the assistance of another person, either as a result of altered consciousness or confusion or by the administration of substances such as carbohydrates or glucagon or other resuscitative actions. On the other hand, the control patient has not suffered severe hypoglycemia events in the last three years. 
All patients have suffered from Type 1 Diabetes Mellitus for at least 20 years, and all were 60 years or older, thus differentiated only by severe hypoglycemia events. The data were anonymized, identifying the patient by a random ID generated by JCHR. Therefore, it is impossible to identify patients from the data provided.

The surveys in the JCHR database are explained below:

1. Blood Glucose Attitude Scale (BBGAttitudeScale): survey designed to measure the level of fear of hyperglycemia. The assessment consists of 8 items. 

2. Continuous Glucose Monitoring (BDataCGM): contains information from the CGM DexCom sensor which is a continuous glucose monitoring system. For the elaboration of this database, each patient has worn this sensor for 7 days, thus saving all the information collected by the sensor in this survey.

3. General patient information (BDemoLifeDiabHxMgmt): contains information on demographics, diabetes management, and history and lifestyle of each patient.

4. Diabetic Arithmetic Test (BDiabNumTest): contains information about the Diabetes Numeracy Test (DNT-15). This test measures numeracy skills in patients with diabetes. This test consists of 15 questions in the following areas: nutrition, exercise, blood glucose control, and medication. 

5. Geriatric Depression Scale (BGeriDepressScale): a survey based on the Geriatric Depression Scale. This scale is a 15-item questionnaire based on yes or no questions to participants about how they felt during the last week. This form was specifically designed to identify depression in the elderly and its possible relationship to diabetes. The completion time is approximately 10 minutes.

6. Hypoglycemia Fear Survey (BHypoFearSurvey): contains information on the Hypoglycemia Fear Survey. This survey measures the fear of having a hypoglycemic event among adults with type 1 diabetes. 

7. Assessment of hypoglycemia unawareness (BHypoUnawareSurvey): contains information on the performance of the Clark method. It allows the assessment of hypoglycemia unawareness, and for this purpose, it consists of 8 questions that assess the glycemic threshold and symptomatic responses to hypoglycemia. 

8. Medical Chart (BMedChart): contains information regarding the patient's weight and height, the number of times the patient measures his blood glucose level per day, as well as the number of carbohydrates ingested, or the number of previous hours the patient took food before doing the C-peptide test.


9. Medical condition (BMedicalConditions): contains information on the diseases currently presented by each patient coded with the MedRDA code and the treatment followed for such illnesses (medication, surgery, among others).

10. Medications (BMedication): a database that collects information on each of the medications taken by the different patients and the administered dose of each drug.

11. Montreal Cognitive Assessment (BMoCA): survey based on the Montreal Cognitive Assessment. This assessment is an instrument designed to detect cognitive function. For this purpose, it evaluates the areas of attention and concentration, executive functions, memory, language, visual-constructive skills, conceptual thinking, calculation, and orientation. 

12. Blood tests (BSampleResults): contains information related to blood tests performed on each patient. Specifically, HBAC1, C-peptide, glucose, and creatinine values are collected.

13. Evaluation of different cognitive tests (BTotTestScores): it will collect information on different tests such as:
-Stroke test
-Hopkins Verbal Learning Test
-Symbols and digits test
-Functional Activities Questionnaire
-Groove Board Test
-Vision Assessment
-Duke Social Support Index

# Features

Written in Python

Numpy, scipy and matplotlib as the main libraries for visualization

Pandas as the main library for preprocessing data

Sklearn as a main library for Machine Learning Learning

# Getting started
## Requeriments
To run all the scripts you need the following packages:

Python version 

numpy version

matplotlib version

scipy version

Pandas version


## Running scripts