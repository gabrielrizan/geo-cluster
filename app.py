import io
import base64
import requests
import xml.etree.ElementTree as ET
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

app = Flask(__name__)

def fetch_geo_ids_for_pmid(pmid):
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?"
        f"dbfrom=pubmed&db=gds&linkname=pubmed_gds&id={pmid}&retmode=xml"
    )
    response = requests.get(url)
    geo_ids = []
    if response.status_code == 200:
        try:
            root = ET.fromstring(response.text)
            for link in root.findall(".//Link"):
                geo_id = link.find("Id").text
                geo_ids.append(geo_id)
        except ET.ParseError:
            print(f"Error parsing XML for PMID {pmid}")
    else:
        print(f"Failed to fetch GEO IDs for PMID {pmid}")
    return geo_ids


def fetch_geo_details(geo_id):
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
        f"db=gds&id={geo_id}&retmode=xml"
    )
    response = requests.get(url)
    details = {}
    if response.status_code == 200:
        try:
            root = ET.fromstring(response.text)
            docsum = root.find(".//DocSum")
            if docsum is not None:
                for item in docsum.findall("Item"):
                    name = item.attrib.get("Name")
                    if name == "title":
                        details["title"] = item.text or ""
                    elif name == "summary":
                        details["summary"] = item.text or ""
                    elif name == "organism":
                        details["organism"] = item.text or ""
                    elif name == "type":
                        details["experiment_type"] = item.text or ""
                    elif name == "overall_design":
                        details["overall_design"] = item.text or ""
        except ET.ParseError:
            print(f"Error parsing GEO details for GEO ID {geo_id}")

    for key in ["title", "experiment_type", "summary", "organism", "overall_design"]:
        if key not in details:
            details[key] = ""
    return details


def get_dataset_description(details):
    return " ".join(
        [
            details.get("title", ""),
            details.get("experiment_type", ""),
            details.get("summary", ""),
            details.get("organism", ""),
            details.get("overall_design", ""),
        ]
    )


def cluster_datasets(dataset_texts, n_clusters=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(dataset_texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    X_reduced = pca.fit_transform(X.toarray())

    return X_reduced, labels


def create_plot(X_reduced, labels, geo_ids, pmid_mapping):
    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="viridis", s=50)

    for i, geo_id in enumerate(geo_ids):
        associated_pmids = ", ".join(pmid_mapping.get(geo_id, []))
        annotation = f"GEO: {geo_id}\nPMIDs: {associated_pmids}"
        plt.annotate(annotation, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=8)

    plt.title("GEO Dataset Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return image_base64


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pmids_text = request.form.get("pmids", "")
        pmids = [line.strip() for line in pmids_text.splitlines() if line.strip().isdigit()]

        geo_dataset_map = {}

        for pmid in pmids:
            geo_ids = fetch_geo_ids_for_pmid(pmid)
            for geo_id in geo_ids:
                geo_dataset_map.setdefault(geo_id, []).append(pmid)

        dataset_texts = []
        geo_ids_list = []
        for geo_id in geo_dataset_map:
            details = fetch_geo_details(geo_id)
            description = get_dataset_description(details)
            if description.strip():
                dataset_texts.append(description)
                geo_ids_list.append(geo_id)

        if not dataset_texts:
            return "No GEO datasets found for provided PMIDs."

        X_reduced, labels = cluster_datasets(dataset_texts)

        image_base64 = create_plot(X_reduced, labels, geo_ids_list, geo_dataset_map)

        return render_template("result.html", image_base64=image_base64)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
