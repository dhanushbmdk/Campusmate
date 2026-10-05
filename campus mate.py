import os
import random
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK Downloads
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# ---------------- COLLEGE DATA ----------------

ug_courses = [
    "B.Sc AI & Data Science", "B.Sc Computer Science", "BCA", "B.Sc Biotechnology",
    "B.Sc Microbiology", "B.Sc Chemistry", "B.Com", "B.Com (CA)", "B.A Tamil",
    "B.A English", "B.A History", "BBA", "BBA (CA)", "B.Sc Electronics & Communication",
    "B.Sc Maths", "B.Sc Physics", "B.Sc Geology"
]

pg_courses = [
    "M.Sc (Computer Science)", "M.Sc Chemistry", "M.Sc Biotechnology",
    "M.Sc Maths", "M.Com", "M.A English"
]

departments = {
    "AI & Data Science": "HOD: A.Praveen \nContact: 6380100255",
    "Computer Science": "HOD: A.Praveen \nContact: 6380100255",
    "BCA": "HOD: A.Praveen \nContact: 6380100255",
    "Commerce": "HOD: Saravanan\nContact: 9865787766",
    "Commerce (CA)": "HOD: Ramasamy",
    "Electronics & Communication": "HOD: Manikandan\nContact: 9003640740",
    "History": "HOD: Selvaraj\nContact: 9443955875",
    "English": "HOD: Sivakumar\nContact: 8940464484",
    "Tamil": "HOD: Durga\nContact: 7871285087",
    "Maths": "HOD: Nithya\nContact: 9944643035",
    "Biotechnology": "HOD: saranya\nContact: 8220101219",
    "Microbiology": "HOD: saranya\nContact: 8220101219",
    "BBA": "HOD: Murugasan\nContact: 9698199746",
}

library_info = """🏛️ LIBRARY INFORMATION
Block: Main Block | Ground Floor | Room No: 3
Working Hours: 9:45 AM - 3:45 PM

Facilities:
• Text Books & Reference Books
• Journals & Previous Year Question Papers
• Digital Library & Free Internet Facility"""

hostel_info = """🏠 HOSTEL FACILITIES
• Boys Hostel: Sri Ganesh College of Engineering, Mettupatty
• Girls Hostel: Sri Ganesh College of Arts & Science, Ammapet

Facilities:
• Furnished Rooms, Hygienic Food & RO Water
• Study Area & High-Speed Wi-Fi

Contact Hostel Office for admission & allotment."""

canteen_info = """🍔 CANTEEN & CAFETERIA
Available Items: Breakfast, Lunch, Snacks, Tea/Coffee, Cool Drinks, Ice Cream
Timings: 9:30 AM - 4:45 PM"""

bus_routes = {
    "24": ["College", "Ganesh Engineering college mettupatti", "Vazhapadi", "Athunurpatti", "Karumandurai"],
    "30": ["College", "Salem Junction", "Thoppur", "Tharamangalam"],
    "34": ["College", "Seelanaickenpatti", "Sivathapuram"],
    "32": ["College", "Kondalampatti", "Magudanchavadi", "Edappadi"],
    "21": ["College", "Veeranam", "Chinnanur", "Valasur"],
    "17": ["College", "Ayothiapattinam", "Achankuttapatti", "Bommidi"],
}

labs = """💻 LABORATORY FACILITIES
Labs: Programming Lab, Data Science Lab, Internet Lab, CS Lab
Facilities: High-Speed Computers, Projectors, Software Applications & Internet"""

sports_info = """⚽ SPORTS FACILITIES
Outdoor & Indoor: Cricket, Football, Volleyball, Basketball, Badminton, Athletics
Spacious Sports Ground & Indoor Facilities Available."""

admission_info = """📋 ADMISSION PROCEDURE
Required Certificates:
1. 10th & 12th Mark Sheets
2. Transfer Certificate (TC) & Conduct Certificate
3. Community Certificate
4. Passport Size Photos

Contact College Admission Office for details."""

fees_info = """💰 FEE STRUCTURE (Academic Year 2026 - 2027)

UG Courses Semester Fees:
• B.Sc AI & DS / CS / BCA: ₹18,000 / Semester
• B.Sc Biotech / Micro / Chem / B.Com / B.Com(CA): ₹17,000 / Semester
• B.A Tamil / Eng / Hist / BBA / ECE / Maths / Physics / Geology: ₹16,000 / Semester

PG Courses Semester Fees:
• M.Sc CS: ₹19,000 / Semester
• M.Sc Chem / Biotech / M.Com: ₹18,000 / Semester
• M.A Eng / M.Sc Maths: ₹17,000 / Semester

Additional Charges:
• Admission & Application Cost: UG ₹1,300 | PG ₹1,500
• Uniform & Books: UG ₹1,500/Year | PG ₹2,000/Year
• Hostel Fees: Rent & Mess ₹46,000/Year (Deposit: ₹3,000)"""

exam_info = """📝 EXAMINATION CELL
Services: Internal Assessments, Model Exams, Semester Exams, Practical Schedule, Hall Tickets & Results."""

principal_info = """🎓 PRINCIPAL OFFICE
Principal: Dr.S.Senthilkumar
Contact Office Desk for Appointments."""


# ---------------- NLP INTENT MAPPING ----------------

intents = {
    "greeting": {
        "patterns": ["hi", "hello", "hai", "hey", "good morning", "good evening"],
        "response": "Hello! 👋 Welcome to Sri Ganesh College Information Assistant.\nHow can I help you today?",
    },
    "courses": {
        "patterns": ["what courses are available", "list of degree courses offered", "available departments", "course details"],
        "response": "📚 UG Courses:\n" + "\n".join(["• " + c for c in ug_courses]) + "\n\n🎓 PG Courses:\n" + "\n".join(["• " + c for c in pg_courses]),
    },
    "principal": {
        "patterns": ["who is the principal", "principal contact details", "principal office"],
        "response": principal_info,
    },
    "hod": {
        "patterns": ["hod details", "head of department info", "department heads contact"],
        "response": "👨‍🏫 Department HOD Information:\n\n" + "\n\n".join([f"📌 {dept}\n{details}" for dept, details in departments.items()]),
    },
    "library": {
        "patterns": ["library timing and facilities", "where is the library", "book borrowing"],
        "response": library_info,
    },
    "hostel": {
        "patterns": ["hostel facilities and rooms", "is hostel available", "accommodation details"],
        "response": hostel_info,
    },
    "canteen": {
        "patterns": ["canteen food menu", "food timing canteen", "cafeteria"],
        "response": canteen_info,
    },
    "labs": {
        "patterns": ["computer lab facilities", "laboratories available", "practical lab"],
        "response": labs,
    },
    "sports": {
        "patterns": ["sports ground and games", "athletics basketball cricket"],
        "response": sports_info,
    },
    "admission": {
        "patterns": ["admission procedure", "how to join college", "documents required"],
        "response": admission_info,
    },
    "fees": {
        "patterns": ["fee structure for courses", "how much is tuition fee", "fees payment"],
        "response": fees_info,
    },
    "exam": {
        "patterns": ["examination cell", "semester exam schedule", "hall ticket"],
        "response": exam_info,
    },
    "bus": {
        "patterns": ["college bus routes", "transport details", "bus timing"],
        "response": "🚌 College Bus Routes:\n\n" + "\n\n".join([f"Bus No. {b}\nRoute: {' → '.join(r)}" for b, r in bus_routes.items()]),
    },
    "farewell": {
        "patterns": ["bye", "exit", "thank you", "thanks"],
        "response": "Thank you for using CampusMate! Have a great day 😊",
    },
    "developer": {
        "patterns": ["who created you", "who is your developer", "who made you", "yaru create panna", "developer name", "who created this", "yaru pன்னது"],
        "response": "👨‍💻 I was developed by B Dhanush & J Rithish (Department of B.Sc. (CS) with Artificial Intelligence and Data Science ) (2024-2027)!",
    },

}

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(t) for t in tokens])

training_patterns = []
pattern_to_intent = []

for intent, data in intents.items():
    for pattern in data["patterns"]:
        training_patterns.append(preprocess(pattern))
        pattern_to_intent.append(intent)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(training_patterns)

def get_bot_response(question):
    q_clean = question.lower().strip()
    for bus_no, route in bus_routes.items():
        for stop in route:
            if stop.lower() in q_clean and len(stop) > 3:
                return f"🚌 Bus No. {bus_no}\n\nRoute:\n" + " → ".join(route)

    processed = preprocess(question)
    user_vector = vectorizer.transform([processed])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    best_idx = similarities.argmax()

    if similarities[best_idx] > 0.2:
        return intents[pattern_to_intent[best_idx]]["response"]

    return ("Sorry, I couldn't understand your query.\n\nYou can ask about:\n"
            "• Courses & Fees\n• HOD Contacts\n• Hostel & Canteen\n• Bus Routes\n• Library & Labs")

# ---------------- GUI APPLICATION ----------------

root = tk.Tk()
root.title("CampusMate - Sri Ganesh College Assistant & Grievance Portal")
root.geometry("650x720")
root.configure(bg="#0f172a")

style = ttk.Style()
style.theme_use("default")
style.configure("TNotebook", background="#0f172a", borderwidth=0)
style.configure("TNotebook.Tab", background="#1e293b", foreground="#00f2fe", padding=[15, 6], font=("Segoe UI", 10, "bold"))
style.map("TNotebook.Tab", background=[("selected", "#00f2fe")], foreground=[("selected", "#0f172a")])

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# TAB 1: AI CHATBOT
chat_tab = tk.Frame(notebook, bg="#0f172a")
notebook.add(chat_tab, text="💬 AI Chatbot")

chat_area = scrolledtext.ScrolledText(chat_tab, wrap=tk.WORD, bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 10), bd=0)
chat_area.pack(padx=10, pady=10, fill="both", expand=True)

chat_area.tag_config("user_tag", foreground="#38bdf8", font=("Segoe UI", 10, "bold"))
chat_area.tag_config("bot_tag", foreground="#feca57", font=("Segoe UI", 10))

chat_area.insert(tk.END, "CampusMate Bot:\nWelcome to CampusMate! Ask me about courses, fees, bus routes, library, or hostel facilities! 👋\n\n", "bot_tag")
chat_area.config(state=tk.DISABLED)

input_frame = tk.Frame(chat_tab, bg="#0f172a")
input_frame.pack(fill="x", padx=10, pady=(0, 10))

entry = tk.Entry(input_frame, bg="#1e293b", fg="white", font=("Segoe UI", 11), insertbackground="white", bd=1)
entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 10), ipady=4)

def send_message():
    msg = entry.get().strip()
    if not msg:
        return
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, f"You: {msg}\n\n", "user_tag")
    reply = get_bot_response(msg)
    chat_area.insert(tk.END, f"CampusMate Bot:\n{reply}\n\n", "bot_tag")
    chat_area.config(state=tk.DISABLED)
    entry.delete(0, tk.END)
    chat_area.see(tk.END)

send_btn = tk.Button(input_frame, text="SEND 📩", bg="#00f2fe", fg="#0f172a", font=("Segoe UI", 10, "bold"), bd=0, command=send_message)
send_btn.pack(side=tk.RIGHT, ipadx=10, ipady=3)
root.bind("<Return>", lambda event: send_message())

# TAB 2: COMPLAINT PORTAL
complaint_tab = tk.Frame(notebook, bg="#0f172a")
notebook.add(complaint_tab, text="⚠️ Submit Complaint")

tk.Label(complaint_tab, text="Grievance Redressal Cell", font=("Segoe UI", 14, "bold"), bg="#0f172a", fg="#ff4757").pack(pady=10)

tk.Label(complaint_tab, text="Student Roll No / Name:", bg="#0f172a", fg="#e2e8f0", font=("Segoe UI", 10)).pack(anchor="w", padx=20)
student_entry = tk.Entry(complaint_tab, bg="#1e293b", fg="white", font=("Segoe UI", 11), insertbackground="white")
student_entry.pack(fill="x", padx=20, pady=(0, 10))

tk.Label(complaint_tab, text="Department / Category:", bg="#0f172a", fg="#e2e8f0", font=("Segoe UI", 10)).pack(anchor="w", padx=20)
dept_entry = tk.Entry(complaint_tab, bg="#1e293b", fg="white", font=("Segoe UI", 11), insertbackground="white")
dept_entry.pack(fill="x", padx=20, pady=(0, 10))

tk.Label(complaint_tab, text="Describe your Complaint / Issue:", bg="#0f172a", fg="#e2e8f0", font=("Segoe UI", 10)).pack(anchor="w", padx=20)
complaint_text = tk.Text(complaint_tab, height=8, bg="#1e293b", fg="white", font=("Segoe UI", 10), insertbackground="white")
complaint_text.pack(fill="x", padx=20, pady=(0, 15))

def submit_complaint():
    s_name = student_entry.get().strip()
    s_dept = dept_entry.get().strip()
    s_issue = complaint_text.get("1.0", tk.END).strip()
    
    if not s_name or not s_dept or not s_issue:
        messagebox.showwarning("Warning", "Please fill all details!")
        return
    
    with open("complaints_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Student: {s_name} | Dept: {s_dept}\nIssue: {s_issue}\n{'-'*40}\n")
        
    messagebox.showinfo("Success", "Your complaint has been logged successfully!")
    student_entry.delete(0, tk.END)
    dept_entry.delete(0, tk.END)
    complaint_text.delete("1.0", tk.END)

submit_btn = tk.Button(complaint_tab, text="SUBMIT COMPLAINT 🚀", bg="#ff4757", fg="white", font=("Segoe UI", 11), bd=0, command=submit_complaint)
submit_btn.pack(ipadx=15, ipady=5)

root.mainloop()