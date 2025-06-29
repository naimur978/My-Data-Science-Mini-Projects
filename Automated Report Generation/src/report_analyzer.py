import torch
import spacy
from collections import defaultdict
from tqdm import tqdm

class ReportAnalyzer:
    """
    Class for analyzing medical reports using NLP techniques.
    """
    def __init__(self):
        """Initialize the analyzer with SpaCy model and predefined labels."""
        self.nlp = spacy.load("en_core_web_sm")
        self.labels = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
            "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
            "Support Devices", "No Finding"
        ]
        self.label_synonyms = self._initialize_synonyms()
    
    def _initialize_synonyms(self):
        """Initialize the dictionary of label synonyms."""
        return {
            "Enlarged Cardiomediastinum": ["cardiomediastinal enlargements", "cardiomediastinal enlargement", 
                                         "enlargement of the cardiac silhouette", "shift of the mediastinal structures"],
            "Cardiomegaly": ["enlarged hearts", "enlarged heart", "continued enlargement of the cardiac silhouette", 
                            "moderate cardiomegaly has worsened", "heart size is normal"],
            "Lung Opacity": ["lung densities", "lung density", "pulmonary opacities", "pulmonary opacity"],
            "Lung Lesion": ["pulmonary lesions", "pulmonary lesion"],
            "Edema": ["fluid retentions", "fluid retention", "pulmonary vascular congestion", "vascular engorgement", 
                     "cephalized vasculature consistent with pulmonary venous hypertension"],
            "Consolidation": ["pulmonary consolidations", "pulmonary consolidation", "air-space opacity", 
                            "lobar opacity", "air-space disease", "alveolar filling", "pulmonary dense area", 
                            "lung solidification", "consolidation"],
            "Pneumonia": ["lung infections", "inflammatory lung conditions", "lung infection", 
                         "inflammatory lung condition", "no evidence of pneumonia"],
            "Atelectasis": ["partial lung collapses", "lung collapses", "partial lung collapse", 
                           "lung collapse", "bibasilar atelectatic changes persist", "mild left basal atelectasis"],
            "Pneumothorax": ["collapsed lungs", "air in the chest cavities", "collapsed lung", 
                            "air in the chest cavity", "no definite pneumothorax is seen"],
            "Pleural Effusion": ["fluid in the chest cavities", "pleural fluids", "fluid in the chest cavity", 
                                "pleural fluid", "pleural thickening", "pleural effusions"],
            "Pleural Other": ["other pleural abnormalities", "abnormalities in the chest cavities", 
                            "other pleural abnormality", "abnormalities in the chest cavity"],
            "Fracture": ["bone breaks", "bone fractures", "bone break"],
            "Support Devices": ["medical devices", "right PIC catheter", "central catheter", "endotracheal tube", 
                              "nasogastric tube", "right pigtail catheter", "right pacemaker"],
            "No Finding": ["normal", "lungs are clear", "lungs are grossly clear"]
        }

    def _extract_clauses(self, sent):
        """Extract independent clauses from a sentence."""
        doc = self.nlp(sent)
        clauses = []
        for token in doc:
            if token.dep_ == "ROOT":
                clause = " ".join([t.text for t in token.subtree])
                clauses.append(clause)
        return clauses

    def analyze_reports(self, reports):
        """
        Analyze a list of medical reports and extract observations for each label.
        
        Args:
            reports (list): List of medical report texts
            
        Returns:
            list: List of dictionaries containing observations for each label
        """
        observations = []

        for report in tqdm(reports):
            try:
                doc = self.nlp(report)
            except Exception as e:
                print(f"Error processing report: {report}")
                print(f"Error details: {e}")
                observations.append("No relevant information")
                continue

            report_info = defaultdict(str)
            for sent in doc.sents:
                for label in self.labels:
                    if (label.lower() in sent.text.lower() or 
                        any(synonym in sent.text.lower() for synonym in self.label_synonyms.get(label, []))):
                        clauses = self._extract_clauses(sent.text)
                        if clauses:
                            report_info[label] += " ".join(clauses) + " "
                        else:
                            report_info[label] += sent.text + " "

            observations.append({
                label: report_info[label].strip() or "No relevant information" 
                for label in self.labels
            })

        return observations
