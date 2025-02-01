# 📊 Sys-Stats Dashboard ✨

Welcome to the **Sys-Stats Dashboard**! This project is designed to monitor and visualize system performance in real-time through a sleek and interactive dashboard. Whether you're interested in CPU, RAM, or GPU usage, this tool provides comprehensive insights at your fingertips. 

## 🎢 Features

- **Real-time Monitoring**: Get instant updates on system metrics such as CPU load, memory usage, GPU stats, and essential processes.
  
- **Interactive Dashboards**: Use the web-based or command-line interface for intuitive dashboards, allowing detailed insights into system behavior.
  
- **Cross-Platform Support**: Deployed using Docker, ensuring consistency and easy setup across different operating environments.
  
- **Customizable Refresh Rates**: Adapt the monitoring frequency to suit your needs, enabling faster updates or conserving resources when needed.

- **Ollama API Integration**: Out-of-the-box compatibility with the Ollama API for additional system-specific metrics and insights.

## 🎆 Screenshots

![Web Dashboard](https://raw.githubusercontent.com/obeone/sys-stats/main/docs/web.png)
![CLI Dashboard](https://raw.githubusercontent.com/obeone/sys-stats/main/docs/cli.png)

## 🚀 Getting Started

Ready to get your Sys-Stats Dashboard up and running? Follow these steps:

### 🐳 Running with Docker

#### Requirements

Ensure you have the following installed on your system:

- Docker 🐳
- NVIDIA drivers (if you want GPU monitoring) 🎮

#### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/obeone/sys-stats.git
   cd sys-stats
   `````

1. **Running the System:**

    - **With nvidia GPU Support:**

    ```bash
    docker compose -f compose.yaml -f compose.gpu.yaml up -d
    ```

    - **Without nvidia GPU Support:**
  
    ```bash
    docker compose up -d
    ```

   The service will be running at `http://localhost:5000`.

### ⚙️ Running Without Docker

If you prefer to run the Sys-Stats Dashboard directly on your machine without using Docker, follow these steps for a seamless setup:

#### Prerequisites

Before setting up the project, make sure you have the following prerequisites installed:

- **Python 3.12+**: You'll need Python to run the application natively.
- **pip**: Python's package manager to install required dependencies.
- **NVIDIA drivers**: For GPU monitoring (if applicable).

#### Setup Instructions

1. **Clone the Repository:**

   Begin by cloning the repository to your local machine:

   ```bash
   git clone https://github.com/obeone/sys-stats.git
   cd sys-stats
   ```

2. **Create a Virtual Environment:**

   It's a good practice to use a virtual environment to manage dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   Install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   If you're using the Ollama API, ensure you set the `OLLAMA_API_URL` environment variable:

   ```bash
   export OLLAMA_API_URL="http://localhost:11434"
   ```

5. **Start the Application:**

   Run the Flask application:

   ```bash
   python app.py
   ```

   The application will start on `http://localhost:5000`.

## 📺 Using the CLI

To use the CLI for live monitoring, execute:

```bash
python cli-v2.py --url http://localhost:5000/stats --interval 5
```

This command launches the CLI with a 5-second refresh interval.

You're all set! Now you can enjoy using the Sys-Stats Dashboard on your local setup without Docker.

## 🧑‍💻 Contributing

We welcome contributions from the community! Feel free to open issues, suggest features, or submit pull requests. Let's build a better Sys-Stats Dashboard together!

## 📝 Notes

This repo is clearly messy, but it was supposed to be only for my own use!
