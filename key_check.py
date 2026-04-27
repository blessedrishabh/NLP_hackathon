import google.generativeai as genai
genai.configure(api_key="AIzaSyDqkfh848xHjcnEbqK45NHMz73Z-R-nFFE")
for model in genai.list_models():
    print(model)