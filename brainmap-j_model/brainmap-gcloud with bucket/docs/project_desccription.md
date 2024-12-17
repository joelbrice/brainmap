<!-- Tumors, MRI image recognition and AI -->

<!-- the Team -->

Gökhan Dede: add short bio here

Antonio Ferri: add short bio here

Markus Kaller: add short bio here

Mosleh Rostami: add short bio here

Joël Brice Voumo Tiogo – Project Lead: add short bio here


<!-- Project outline -->

Objective

The primary aim of this project is to develop an AI-based tool capable of accurately identifying whether a patient has a brain tumor and determining its specific subtype. Differentiating between tumor subtypes is critical for tailoring treatment recommendations, as these classifications directly influence clinical decision-making. Brain tumor diagnosis is complex and often necessitates specialized expertise, which may not be accessible to all patients in a timely manner. This tool seeks to bridge that gap by providing an accessible and precise diagnostic aid.


Structure

With a focus on addressing the challenges of subtype differentiation, we structured the project as follows:

Data Collection: Assemble a comprehensive dataset of MRI scans that includes several classes of brain tumor subtypes (e.g. pituitary, meningioma and glioma), as well as scans of healthy brains. The dataset should encompass all relevant features needed for AI to distinguish between different tumor types. Preprocessing may be required to ensure the data is algorithm-ready.

Model Development: Select, build, and train a CNN model tailored for the classification task. The model will serve as the core of the AI tool, capable of recognizing and differentiating tumor subtypes.

Prediction and Diagnosis: Develop the model to reliably predict the presence of a tumor and its subtype in unseen MRI scans. As a high percentage of gliomas are clinically classified as malignant, the prediction
will serve as an indirect estimator of tumor malignancy.
reduction will be an indirect estimator of tumor malignancy.
Production Phase (Optional): If the model achieves the desired performance, create a web-based AI tool to present the results in an accessible format. A chat-based framework incorporating natural language processing (NLP) is planned to make the tool user-friendly.

Timeline

Initial Model Development: Build and train a basic model locally (2 days).
Model Evaluation: Select the best-performing model and conduct a collective review (1 day).
Production Integration: Begin the production phase while simultaneously refining the model and incorporating XAI features (
Frontend Development: Develop a user-friendly web interface (2-3 days). days). The improved model will replace the initial version.
