import uvicorn



# if __name__ == "__main__":
uvicorn.run("umls_services:app", host="127.0.0.1", port=5000, log_level="info")
# uvicorn.run("relation_reader:app", host="127.0.0.1", port=5000, log_level="info")