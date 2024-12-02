Tenure Tracker

Welcome to Tenure Tracker, a project designed to provide valuable insights into faculty hiring trends at higher education institutions in the United States. This repository aims to compile data, visualize trends, and assist academics in better understanding the faculty hiring landscape over multiple years.

Overview

The Tenure Tracker project gathers, processes, and presents data on faculty hiring at higher education institutions. This project was inspired by the desire to understand the long-term trends in academic hiring across various disciplines.

Goals of the Project:

Collect and organize publicly available data on faculty hires.

Provide tools for visualizing and analyzing trends over time.

Create a resource for early-career academics interested in understanding the faculty hire landscape.

Features

Data Collection: Scripts for collecting information from various academic sources, including department websites, public announcements, and databases.

Data Visualization: Tools for visualizing trends such as hiring rates by field, gender diversity, and changes over time.

Analysis and Reporting: Jupyter notebooks and scripts that facilitate deep dives into the data, offering insights into specific disciplines and institutions.

Repository Structure

data-collection: Contains raw and processed datasets related to the number of faculty in the US over the years and economic features (these are listed and explained in "data_description_and_units.csv").

data-analysis: Includes the final analysis Jupyter notebook with the results. 

api: Contains the required py files that feeds into the streamlit app.

dump-ignore: older files and folders. 


Getting Started

To get started with the project, follow these steps:

Clone the Repository:

git clone https://github.com/sadityag/tenure-tracker.git
cd tenure-tracker

Install Dependencies:
Make sure you have Python installed, then install the required packages using pip:

pip install -r requirements.txt


Explore the data: Use our public app at https://tenure-tracker.streamlit.app/


Run the Analysis: Use the provided Jupyter notebook named "final_analysis" in "data-analysis" folder.


Requirements

Python 3.8+
Jupyter Notebook
Pandas, NumPy, Matplotlib, and other dependencies listed in requirements.txt
