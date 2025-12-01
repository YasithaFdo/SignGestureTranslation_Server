# SignGestureTranslation_Server

## Project Structure Overview

* server.py — Runs the actual backend server. This is the main entry point of the application.
* gpt.py — A standalone text generator used only for testing the GPT model independently from the server.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YasithaFdo/SignGestureTranslation_Server.git
```

### 2. Create the Models Folder

Create a folder named **Models** inside the project directory.
Place your trained models inside this folder.

### 3. Update Configuration

* Change the **categories list** based on your project requirements.
* Update the **file paths** in the code to match your environment.

### 4. Run the Server

```bash
py server.py
```

Or on Linux/macOS:

```bash
python3 server.py
```
