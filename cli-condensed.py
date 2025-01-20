#!/usr/bin/env python3

import requests
import time
import argparse
import shutil
from termcolor import colored
import os
from datetime import datetime, timedelta, timezone
from prettytable import PrettyTable

SYS_STATS_API_URL = os.getenv('SYS_STATS_API_URL', 'http://localhost:5000/stats')

def fetch_stats(api_url):
    """
    Fetch statistics from the specified API URL.
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(colored(f"Error fetching stats: {e}", "red"))
        return None

def human_readable_size(size):
    """
    Convert bytes into a human-readable format.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"

def time_until(expiration):
    """
    Calculate the time remaining until a given expiration datetime.
    """
    exp_time = datetime.fromisoformat(expiration.replace('Z', '+00:00')).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    delta = exp_time - now
    if delta.total_seconds() <= 0:
        return "Expired"
    return str(timedelta(seconds=int(delta.total_seconds())))

def truncate_cmdline(cmdline, width):
    """
    Truncate command line string if it exceeds a specified width.
    """
    return cmdline if len(cmdline) <= width else cmdline[:width - 3] + "..."

def clear_terminal():
    """
    Clear the terminal screen.
    """
    print("\033[H\033[J", end="")

def display_summary(data):
    """
    Display a summary in a single line to occupy horizontal space better.
    """
    current_time_str = colored(data["current_time"], "white", attrs=["bold"])
    cpu_str = colored(f"{data['cpu']:.1f}%", "green", attrs=["bold"])
    ram_percent_str = colored(f"{data['ram']['percent']:.1f}%", "yellow", attrs=["bold"])
    ram_total_str = colored(human_readable_size(data['ram']['total']), "cyan")

    summary_parts = [
        f"Time: {current_time_str}",
        f"CPU: {cpu_str}",
        f"RAM: {ram_percent_str} / {ram_total_str}",
    ]

    if data.get("has_gpu") and data.get("gpu"):
        gpu_data = data['gpu'][0]
        gpu_usage_str = colored(f"{gpu_data['load']:.1f}%", "blue", attrs=["bold"])
        gpu_vram_str = colored(f"{gpu_data['memoryPercent']:.1f}%", "magenta", attrs=["bold"])
        summary_parts.append(f"GPU: {gpu_usage_str}")
        summary_parts.append(f"VRAM: {gpu_vram_str}")

    summary_line = " | ".join(summary_parts)

    print(colored("Summary:", "cyan", attrs=["bold"]))
    print(summary_line)

def build_table_top_cpu(data, width):
    """
    Build PrettyTable for top CPU processes, limited to `width` in total.
    Returns the string representation of the table (not printed yet).
    """
    table = PrettyTable()
    table.field_names = ["PID", "Name", "CPU %", "Cmdline"]

    # On essaie de répartir la largeur pour les colonnes
    nb_columns = len(table.field_names)
    width_per_col = max((width - 4) // nb_columns, 5)
    table._max_width = {field: width_per_col for field in table.field_names}

    for field in table.field_names:
        table.align[field] = "l"

    for proc in data["top_cpu"]:
        table.add_row([
            colored(proc['pid'], "cyan"),
            colored(proc['name'], "green"),
            colored(f"{proc['cpu_percent']:.1f}%", "yellow", attrs=["bold"]),
            colored(truncate_cmdline(proc['cmdline'], width_per_col), "white")
        ])

    return str(table)

def build_table_top_memory(data, width):
    """
    Build PrettyTable for top Memory processes, limited to `width` in total.
    Returns the string representation of the table (not printed yet).
    """
    table = PrettyTable()
    table.field_names = ["PID", "Name", "Mem Usage", "Mem %", "Cmdline"]

    nb_columns = len(table.field_names)
    width_per_col = max((width - 4) // nb_columns, 5)
    table._max_width = {field: width_per_col for field in table.field_names}

    for field in table.field_names:
        table.align[field] = "l"

    for proc in data["top_memory"]:
        table.add_row([
            colored(proc['pid'], "cyan"),
            colored(proc['name'], "green"),
            colored(human_readable_size(proc['memory_usage']), "blue"),
            colored(f"{proc['memory_percent']:.1f}%", "yellow", attrs=["bold"]),
            colored(truncate_cmdline(proc['cmdline'], width_per_col), "white")
        ])

    return str(table)

def build_table_gpu(data, width):
    """
    Build PrettyTable for GPU stats.
    """
    table = PrettyTable()
    table.field_names = ["Name", "Load (%)", "VRAM", "VRAM (%)", "Fan (%)", "Power (W)", "Temp (°C)"]

    nb_columns = len(table.field_names)
    width_per_col = max((width - 4) // nb_columns, 5)
    table._max_width = {field: width_per_col for field in table.field_names}

    for field in table.field_names:
        table.align[field] = "l"

    for gpu in data['gpu']:
        table.add_row([
            colored(gpu['name'], "green", attrs=["bold"]),
            colored(f"{gpu['load']:.1f}%", "blue"),
            colored(human_readable_size(gpu['memoryUsed']), "magenta"),
            colored(f"{gpu['memoryPercent']:.1f}%", "magenta"),
            colored(f"{gpu['fanSpeed']:.1f}%", "cyan"),
            colored(f"{gpu['powerDraw']:.1f} W", "yellow"),
            colored(f"{gpu['temperature']:.1f}°C", "red", attrs=["bold"])
        ])

    return str(table)

def build_table_ollama(data, width):
    """
    Build PrettyTable for Ollama stats.
    """
    table = PrettyTable()
    table.field_names = ["Model Name", "VRAM Usage", "Expiration"]

    nb_columns = len(table.field_names)
    width_per_col = max((width - 4) // nb_columns, 5)
    table._max_width = {field: width_per_col for field in table.field_names}
    for field in table.field_names:
        table.align[field] = "l"

    for model in data["ollama_processes"]["models"]:
        table.add_row([
            colored(model['name'], "green", attrs=["bold"]),
            colored(human_readable_size(model['size_vram']), "blue"),
            colored(time_until(model['expires_at']), "yellow")
        ])

    return str(table)

def table_str_width(table_str):
    """
    Return the maximum line width of a table string (split by lines).
    """
    lines = table_str.splitlines()
    return max((len(line) for line in lines), default=0)

def merge_tables_side_by_side(t1, t2, spacing=4):
    """
    Fusionne horizontalement les 2 tableaux (sous forme de strings) et
    renvoie la liste de lignes du résultat.
    """
    lines1 = t1.splitlines()
    lines2 = t2.splitlines()
    max_len1 = max(len(line) for line in lines1) if lines1 else 0
    max_len2 = max(len(line) for line in lines2) if lines2 else 0

    max_lines = max(len(lines1), len(lines2))
    merged_lines = []
    for i in range(max_lines):
        part1 = lines1[i] if i < len(lines1) else ""
        part2 = lines2[i] if i < len(lines2) else ""
        # on pad part1 pour aligner
        part1 = part1.ljust(max_len1)
        merged_lines.append(part1 + " " * spacing + part2)
    return merged_lines

def display_tables_side_by_side_or_stacked(table_str_1, table_str_2, total_width, title_1, title_2):
    """
    Affiche 2 tableaux soit côte à côte si la largeur le permet, soit empilés.
    - total_width: largeur totale du terminal
    """
    w1 = table_str_width(table_str_1)
    w2 = table_str_width(table_str_2)
    spacing = 4
    needed = w1 + w2 + spacing

    # Titre 1
    print(colored(f"\n{title_1}:", "cyan", attrs=["bold", "underline"]))
    # Titre 2 (on l'affichera juste au-dessus du second tableau, ou au-dessus du bloc fusionné)
    # Si on fusionne, on peut mettre un seul titre, ou deux. Ici on va simplement mettre le deuxième titre après un saut de ligne.
    # On va d'abord décider si on fusionne ou pas.

    if needed <= total_width:
        # On fait la fusion
        # On affiche un titre combiné
        # (pour la démo, on fait un double titre sur une ligne, question de goût)
        print(colored(f"{title_2}:", "cyan", attrs=["bold", "underline"]), "(côte à côte)")
        merged_lines = merge_tables_side_by_side(table_str_1, table_str_2, spacing=spacing)
        for line in merged_lines:
            print(line)
    else:
        # Trop large => on affiche l'un dessous l'autre
        print(table_str_1)
        print(colored(f"\n{title_2}:", "cyan", attrs=["bold", "underline"]))
        print(table_str_2)

def main():
    parser = argparse.ArgumentParser(description="CLI for Server Stats Dashboard")
    parser.add_argument("--url", type=str, default=SYS_STATS_API_URL, help="API URL for stats endpoint")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--oneline", action='store_true', help="Display stats on a single line with icons.")
    args = parser.parse_args()

    while True:
        clear_terminal()
        terminal_width, _ = shutil.get_terminal_size(fallback=(80, 20))
        stats = fetch_stats(args.url)
        if stats:
            # Soit on affiche "oneline", soit un résumé + tableaux
            if args.oneline:
                # Oneline
                cpu_icon = "💻"
                ram_icon = "🧠"
                gpu_icon = "🎮"

                cpu_usage_str = colored(f"{stats['cpu']:.1f}%", "green", attrs=["bold"])
                ram_usage_str = colored(f"{stats['ram']['percent']:.1f}%", "yellow", attrs=["bold"])
                ram_total_str = colored(human_readable_size(stats['ram']['total']), "cyan")

                usage_line = (
                    f"{cpu_icon} CPU: {cpu_usage_str}  "
                    f"{ram_icon} RAM: {ram_usage_str} / {ram_total_str}"
                )

                if stats.get("has_gpu") and stats.get("gpu"):
                    gpu_data = stats['gpu'][0]
                    gpu_usage_str = colored(f"{gpu_data['load']:.1f}%", "blue", attrs=["bold"])
                    vram_usage_str = colored(f"{gpu_data['memoryPercent']:.1f}%", "magenta", attrs=["bold"])
                    usage_line += f"  {gpu_icon} GPU: {gpu_usage_str} VRAM: {vram_usage_str}"

                print(colored("\nOneline Stats:", "cyan", attrs=["bold"]))
                print(usage_line)
            else:
                # On affiche le summary sur une ligne
                display_summary(stats)

                # Construire deux tableaux : top_cpu et top_memory
                table_cpu_str = build_table_top_cpu(stats, width=terminal_width // 2)
                table_mem_str = build_table_top_memory(stats, width=terminal_width // 2)

                # Les afficher soit côte à côte si possible, soit stackés
                display_tables_side_by_side_or_stacked(
                    table_cpu_str, table_mem_str,
                    total_width=terminal_width,
                    title_1="Top CPU Processes",
                    title_2="Top Memory Processes"
                )

                # GPU et Ollama : s'ils existent
                if stats.get("has_gpu") and stats.get("gpu"):
                    table_gpu_str = build_table_gpu(stats, width=terminal_width // 2)
                else:
                    table_gpu_str = None

                if stats.get("ollama_processes", {}).get("models"):
                    table_ollama_str = build_table_ollama(stats, width=terminal_width // 2)
                else:
                    table_ollama_str = None

                # Affichage GPU + Ollama côté à côte si possible
                # On ne les affiche que s'il y a de la data
                if table_gpu_str and table_ollama_str:
                    print()  # saut de ligne
                    display_tables_side_by_side_or_stacked(
                        table_gpu_str, table_ollama_str,
                        total_width=terminal_width,
                        title_1="GPU Stats",
                        title_2="Ollama Stats"
                    )
                elif table_gpu_str:
                    print(colored("\nGPU Stats:", "cyan", attrs=["bold", "underline"]))
                    print(table_gpu_str)
                elif table_ollama_str:
                    print(colored("\nOllama Stats:", "cyan", attrs=["bold", "underline"]))
                    print(table_ollama_str)

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
