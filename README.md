# Server Stats Dashboard

## Overview

This is a server stats dashboard application that provides real-time monitoring of system statistics. The application consists of three main components:

1. **Server-side Flask Application (`app.py`)**: Handles the API endpoints for fetching system statistics.
2. **Web Interface (`index.html`)**: A user-friendly web interface to visualize the system stats in a browser.
3. **Command Line Interface (CLI) (`cli.py`)**: A command-line tool to fetch and display system statistics.

## Project Structure

```
│── sys-stats/
│   ├── app.py
│   ├── index.html
│   └── cli.py
└── README.md
```

## Components

### 1. `app.py` - Server-side Flask Application

- **Purpose**: This is the backend server that exposes an API endpoint to fetch system statistics.
- **Dependencies**: Flask (a Python web framework).

#### Usage

To start the server, navigate to the project directory and run:

```sh
python3 app.py
```

By default, the server runs on `http://localhost:5000`.

### 2. `index.html` - Web Interface

- **Purpose**: This is a web page that fetches system statistics from the API endpoint and displays them in real-time.
- **Dependencies**: None (pure HTML/CSS/JavaScript).

#### Usage

Open `index.html` in a web browser to view the live stats dashboard. You can also access it through a server by placing `index.html` on a web server.

### 3. `cli.py` - Command Line Interface (CLI)

- **Purpose**: This is a command-line tool that fetches and displays system statistics.
- **Dependencies**: Python libraries such as `requests`, `prettytable`, `shutil`, `termcolor`, and `datetime`.

#### Usage

To use the CLI, navigate to the project directory and run:

```sh
python3 cli.py --url http://localhost:5000/stats --interval 5
```

- `--url`: The URL of the API endpoint (default is `http://localhost:5000/stats`).
- `--interval`: The refresh interval in seconds (default is 5 seconds).
- `--oneline`: Display stats on a single line with icons.

## Environment Variables

You can set the `SYS_STATS_API_URL` environment variable to specify the API URL if it's different from the default:

```sh
export SYS_STATS_API_URL=http://your-custom-api-url/stats
```

## Contributing

Feel free to contribute by submitting pull requests or issues. Any improvements or bug fixes are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The project uses various Python libraries to fetch and display system statistics.
- Inspired by real-time monitoring tools, this dashboard aims to provide a simple yet effective way to monitor server performance.
