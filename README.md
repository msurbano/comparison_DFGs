# A Tool for Visual Insights Search from Collections of Directly-Follows Graphs

Process mining is a discipline that enables the analysis of business processes from event logs. The Directly-Follows Graph (DFG) is one of the most used visualization types employed in this domain. However, the extraction of valuable information from DFGs requires significant manual effort from users due to the limitations of current process mining tools. To address this challenge, we propose a visual tool designed to visually compare several DFGs. The tool proposed has been developed with Streamlit (https://streamlit.io/), which is a framework that enables the conversion of data Python scripts into shareable web applications.

## Running the tool Locally

To run the tool locally, follow these steps:
1. Make sure you have Python 3.8+ installed, as well as `graphviz` on your system.
2. Clone the repository to your local machine.
3. Open a terminal and navigate to the root directory of the repository.
4. Install the dependencies:
`pip install -r requirements.txt`
5. Run the following command to launch the application:
`python -m streamlit run 1_Data_Context.py`

## Access via Web

It is available at https://comparisondfgs.streamlit.app/
