import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')


model = pickle.load(open('model.pkl', 'rb'))
TV = pickle.load(open('TV.pkl', 'rb'))



def cleanRes(txt_list):
    cleaned_list = []
    for txt in txt_list:
        cleantxt = re.sub(r'http\S+', ' ', txt)
        cleantxt = re.sub('RT|cc', ' ', cleantxt )
        cleantxt = re.sub(r'#', ' ', cleantxt )
        cleantxt = re.sub(r'@\S+', ' ', cleantxt )
        cleantxt = re.sub('[%s]' % re.escape(r'''!"#$%&'()*+,-./:;<>=?@[\]^_`{|}~'''), ' ', cleantxt)
        cleantxt = re.sub(r'[^\x00-\x7f]' , ' ' , cleantxt )
        cleantxt = cleantxt.lower()
        cleantxt = cleantxt.strip()
        cleantxt = re.sub(r'\s+', ' ', cleantxt )
        cleantxt = re.sub(r'\n+', ' ', cleantxt )
        cleaned_list.append(cleantxt)
    return cleantxt

def main():
  st.title("Resume Screening App")
  upload_resume = st.file_uploader("Upload Resume" , type=['txt', 'pdf'])


  if upload_resume is not None:
      try:
        resume_bytes = upload_resume.read()
        resume_text = resume_bytes.decode('utf-8')
      except UnicodeDecodeError:
        resume_text = resume_bytes.decode('latin-1')  

      cleaned_res = cleanRes([resume_text])
      features = TV.transform([cleaned_res])
      predict_id = model.predict(features)[0]
      

      category_mapping = {
      15: "Java Developer",
      23: "Testing",
      8: "DevOps Engineer",
      20: "Python Developer",
      24: "Web Designing",
      12: "HR",
      13: "Hadoop",
      3: "Blockchain",
      10: "ETL Developer",
      18: "Operations Manager",
      6: "Data Science",
      22: "Sales",
      16: "Mechanical Engineer",
      1: "Arts",
      7: "Database",
      11: "Electrical Engineering",
      14: "Health and fitness",
      19: "PMO",
      4: "Business Analyst",
      9: "DotNet Developer",
      2: "Automation Testing",
      17: "Network Security Engineer",
      21: "SAP Developer",
      5: "Civil Engineer",
      0: "Advocate",}

      category_name = category_mapping.get(predict_id,"Unknown")
      st.write("Predicted Category : ",category_name)


if __name__ == "__main__":
    main()