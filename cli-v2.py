#!/usr/bin/env python3

import requests
import time
import argparse
import shutil
import os
import threading
from datetime import datetime, timedelta, timezone
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
import readchar  # Pour capturer les frappes de touches

# URL de l'API par défaut
SYS_STATS_API_URL = os.getenv('SYS_STATS_API_URL', 'http://localhost:5000/stats')

console = Console()

# États de la dashboard
is_paused = False
show_help_flag = False
refresh_interval = 5  # Intervalle de rafraîchissement par défaut en secondes

# Verrous pour les opérations thread-safe
state_lock = threading.Lock()
stats_lock = threading.Lock()

# Événements pour la synchronisation
exit_event = threading.Event()
rebuild_layout_event = threading.Event()

# Variable partagée pour stocker les dernières statistiques
latest_stats = None


def fetch_stats(api_url):
    """Récupère les statistiques depuis l'URL de l'API spécifiée."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Erreur lors de la récupération des stats:[/bold red] {e}")
        return None


def human_readable_size(size):
    """Convertit des octets en un format lisible."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def time_until(expiration):
    """Calcule le temps restant jusqu'à une date d'expiration donnée."""
    try:
        exp_time = datetime.fromisoformat(expiration.replace('Z', '+00:00')).astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        delta = exp_time - now
        if delta.total_seconds() <= 0:
            return "[bold red]Expiré[/bold red]"
        return str(timedelta(seconds=int(delta.total_seconds())))
    except Exception:
        return "N/A"


def truncate_cmdline(cmdline, width):
    """Tronque la chaîne de commande si elle dépasse une largeur spécifiée."""
    return cmdline if len(cmdline) <= width else cmdline[:width - 1] + "…"


def truncate_name(name, max_length=15):
    """Tronque le nom si il dépasse une longueur maximale spécifiée."""
    return name if len(name) <= max_length else name[:max_length - 1] + "…"


def create_layout():
    """Crée la mise en page initiale pour le dashboard."""
    layout = Layout()

    # Divise la mise en page principale en en-tête et corps
    layout.split(
        Layout(name="header", size=1),
        Layout(name="body")
    )

    # Divise le corps en sections supérieure et inférieure
    layout["body"].split(
        Layout(name="upper", ratio=1),
        Layout(name="lower", ratio=1)
    )

    # Divise la section supérieure en résumé et processus
    layout["upper"].split_row(
        Layout(name="summary", ratio=1),
        Layout(name="processes", ratio=2)
    )

    # Divise la section inférieure en processus GPU et statistiques Ollama
    layout["lower"].split_row(
        Layout(name="gpu_processes", ratio=1),
        Layout(name="ollama", ratio=1)
    )

    return layout


def build_header():
    """Construit le panel d'en-tête."""
    header_text = Text("🖥️ Sys Stats", style="bold white on blue")
    return Panel(header_text, height=1, style="blue", padding=(0, 2))


def build_summary(data, interval):
    """Construit le panel de résumé avec l'utilisation du CPU, RAM et GPU."""
    table = Table.grid(expand=True)
    table.add_column(justify="left")
    table.add_column(justify="right")

    current_time = f"[bold]{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}[/bold]"

    table.add_row("Heure actuelle", current_time)
    table.add_row("", "")

    # Informations CPU et RAM
    cpu_usage = f"[bold green]CPU:[/bold green] {data.get('cpu', 0):.1f}%"
    ram_percent = data.get('ram', {}).get('percent', 0)
    ram_total = human_readable_size(data.get('ram', {}).get('total', 0))
    ram_usage = f"[bold yellow]RAM:[/bold yellow] {ram_percent:.1f}% / {ram_total}"
    table.add_row(cpu_usage, ram_usage)

    # Informations GPU
    if data.get("has_gpu") and data.get("gpu"):
        gpu_data = data['gpu'][0]
        gpu_name = truncate_name(gpu_data.get('name', 'N/A'), 15)
        gpu_load = gpu_data.get('load', 0)

        # Calcul du VRAM total
        memory_used = gpu_data.get('memoryUsed', 0)
        memory_percent = gpu_data.get('memoryPercent', 0)
        if memory_percent > 0:
            memory_total = memory_used / (memory_percent / 100)
        else:
            memory_total = 0
        memory_used_str = human_readable_size(memory_used)
        memory_total_str = human_readable_size(memory_total)
        vram_percent = memory_percent

        gpu_title = f"[bold blue]GPU:[/bold blue] {gpu_name} ({memory_total_str})\n"
        table.add_row(gpu_title)

        gpu_info = f"[bold magenta]VRAM:[/bold magenta] {memory_used_str} ({vram_percent:.1f}%)"
        gpu_load_str = f"[bold blue]Charge:[/bold blue] {gpu_load:.1f}%"
        table.add_row(gpu_info, gpu_load_str)

    # Statut (Pause ou en cours)
    with state_lock:
        status = "[bold red]PAUSÉ[/bold red]" if is_paused else ""

    table.add_row("", status)

    summary_panel = Panel(
        table,
        border_style="cyan",
        padding=(0, 1),
        subtitle=f"Taux de rafraîchissement : {interval}s"
    )
    return summary_panel


def build_process_table(processes, key, title):
    """Construit un tableau pour les processus les plus gourmands."""
    if not processes:
        return Panel(
            f"Aucune donnée pour {title}.",
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
    if key == 'top_cpu':
        table.add_column("PID", style="cyan", no_wrap=True, width=6)
        table.add_column("Nom", style="green", width=15)
        table.add_column("CPU%", style="yellow", justify="right", width=6)
        table.add_column("Cmdline", style="white", max_width=20)
    elif key == 'top_memory':
        table.add_column("PID", style="cyan", no_wrap=True, width=6)
        table.add_column("Nom", style="green", width=15)
        table.add_column("Mémoire%", style="blue", justify="right", width=6)
        table.add_column("Cmdline", style="white", max_width=20)

    for proc in processes:
        pid = str(proc.get('pid', 'N/A'))
        name = truncate_name(proc.get('name', 'N/A'))
        cmdline = truncate_cmdline(proc.get('cmdline', ''), 20)
        if key == 'top_cpu':
            cpu_percent = f"{proc.get('cpu_percent', 0):.1f}%"
            table.add_row(pid, name, cpu_percent, cmdline)
        elif key == 'top_memory':
            mem_percent = f"{proc.get('memory_percent', 0):.1f}%"
            table.add_row(pid, name, mem_percent, cmdline)

    return table


def build_processes_panel(data):
    """Construit le panel des processus avec Top CPU et Top Mémoire."""
    top_cpu = data.get('top_cpu', [])
    top_memory = data.get('top_memory', [])

    table_cpu = build_process_table(top_cpu, 'top_cpu', 'CPU')
    table_mem = build_process_table(top_memory, 'top_memory', 'Mémoire')

    processes_table = Table.grid(expand=True)
    processes_table.add_column()
    processes_table.add_column()

    processes_table.add_row(
        Panel(table_cpu, title="[bold cyan]Top CPU[/bold cyan]", border_style="cyan", padding=(0, 1)),
        Panel(table_mem, title="[bold cyan]Top Mémoire[/bold cyan]", border_style="cyan", padding=(0, 1))
    )

    return processes_table


def build_gpu_processes_panel(data):
    """Construit le panel des processus GPU."""
    processes = data.get("top_gpu_processes", [])
    if not processes:
        return Panel(
            "Aucun processus GPU.",
            title="[bold cyan]Processus GPU[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
    table.add_column("PID", style="cyan", no_wrap=True, width=6)
    table.add_column("Nom", style="green", width=15)
    table.add_column("Mémoire Utilisée", style="blue", justify="right", width=10)
    table.add_column("Cmdline", style="white", max_width=20)

    for proc in processes:
        pid = str(proc.get('pid', 'N/A'))
        name = truncate_name(proc.get('name', 'N/A'))
        memory_used = human_readable_size(proc.get('memory_used', 0))
        cmdline = truncate_cmdline(proc.get('cmdline', ''), 20)
        table.add_row(pid, name, memory_used, cmdline)

    return Panel(
        table,
        title="[bold cyan]Processus GPU[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def build_ollama_panel(data):
    """Construit le panel des statistiques Ollama."""
    models = data.get("ollama_processes", {}).get("models", [])
    if not models:
        return Panel(
            "Aucun modèle Ollama.",
            title="[bold cyan]Statistiques Ollama[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
    table.add_column("Modèle", style="green", width=15)
    table.add_column("Taille", style="blue", justify="right", width=10)
    table.add_column("VRAM", style="blue", justify="right", width=10)
    table.add_column("GPU%", style="red", justify="right", width=6)
    table.add_column("Expire", style="yellow", justify="right", width=12)

    for model in models:
        model_name = truncate_name(model.get('name', 'N/A'))
        size_total = human_readable_size(model.get('size', 0))
        size_vram = human_readable_size(model.get('size_vram', 0))
        gpu_loaded_ratio = (model.get("size_vram", 0) / model.get('size', 1)) * 100 if model.get('size', 1) > 0 else 0
        gpu_loaded_str = f"{gpu_loaded_ratio:.0f}%"
        expiration = time_until(model.get('expires_at', ''))
        table.add_row(
            model_name,
            size_total,
            size_vram,
            gpu_loaded_str,
            expiration
        )

    return Panel(
        table,
        title="[bold cyan]Statistiques Ollama[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def build_layout_content(layout, data, interval, terminal_width):
    """Remplit la mise en page avec les données récupérées."""
    # Mise à jour de l'en-tête
    layout["header"].update(build_header())

    # Mise à jour du résumé
    summary_panel = build_summary(data, interval)
    layout["upper"]["summary"].update(summary_panel)

    # Mise à jour des processus (Top CPU et Top Mémoire)
    processes_table = build_processes_panel(data)
    layout["upper"]["processes"].update(processes_table)

    # Mise à jour des processus GPU
    gpu_processes_panel = build_gpu_processes_panel(data)
    layout["lower"]["gpu_processes"].update(gpu_processes_panel)

    # Mise à jour des statistiques Ollama
    ollama_panel = build_ollama_panel(data)
    layout["lower"]["ollama"].update(ollama_panel)


def build_full_screen_help():
    """Construit le panel d'aide en plein écran."""
    help_text = """
[bold yellow]Raccourcis Clavier :[/bold yellow]

[bold green]q[/bold green] - Quitter
[bold green]r[/bold green] - Rafraîchir
[bold green]h[/bold green] - Afficher/Masquer l'aide
[bold green]p[/bold green] - Pause/Reprendre
[bold green]-[/bold green] - Diminuer l'intervalle
[bold green]+[/bold green] - Augmenter l'intervalle

Appuyez de nouveau sur [bold green]h[/bold green] pour revenir.
"""

    help_panel = Panel.fit(
        help_text,
        title="[bold cyan]Aide[/bold cyan]",
        border_style="green",
        padding=(1, 2)
    )
    return help_panel


def keyboard_listener():
    """Écoute les entrées clavier et modifie l'état en conséquence."""
    global is_paused, show_help_flag, refresh_interval, latest_stats
    while not exit_event.is_set():
        key = readchar.readkey()
        with state_lock:
            if key.lower() == 'q':
                exit_event.set()
            elif key.lower() == 'r':
                # Signal pour rafraîchir les données
                rebuild_layout_event.set()  # On veut reconstruire le layout avec les mêmes données
            elif key.lower() == 'h':
                show_help_flag = not show_help_flag
                rebuild_layout_event.set()  # Reconstruire le layout pour afficher/masquer l'aide
            elif key.lower() == 'p':
                is_paused = not is_paused
                rebuild_layout_event.set()  # Reconstruire le layout pour afficher l'état de pause
            elif key == '-':
                if refresh_interval > 1:
                    refresh_interval -= 1
                    rebuild_layout_event.set()  # Reconstruire le layout pour afficher le nouvel intervalle
            elif key == '+':
                if refresh_interval < 60:
                    refresh_interval += 1
                    rebuild_layout_event.set()  # Reconstruire le layout pour afficher le nouvel intervalle


def main():
    """Fonction principale pour exécuter le dashboard CLI des statistiques serveur."""
    global latest_stats
    parser = argparse.ArgumentParser(description="CLI pour le dashboard des statistiques serveur")
    parser.add_argument("--url", type=str, default=SYS_STATS_API_URL, help="URL de l'API pour les statistiques")
    parser.add_argument("--interval", type=int, default=5, help="Intervalle de rafraîchissement en secondes (peut être ajusté avec '+' et '-')")
    args = parser.parse_args()

    global refresh_interval
    with state_lock:
        refresh_interval = args.interval

    layout = create_layout()

    # Démarrer le thread d'écoute des touches clavier
    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    with Live(layout, refresh_per_second=4, screen=True):
        while not exit_event.is_set():
            with state_lock:
                current_interval = refresh_interval
                paused = is_paused
                help_flag = show_help_flag

            if help_flag:
                # Afficher le panel d'aide en plein écran
                help_panel = build_full_screen_help()
                layout.update(help_panel)
            else:
                if not paused:
                    # Récupérer les statistiques si non en pause
                    stats = fetch_stats(args.url)
                    if stats:
                        with stats_lock:
                            latest_stats = stats
                        terminal_size = shutil.get_terminal_size(fallback=(80, 24))
                        terminal_width = terminal_size.columns
                        build_layout_content(layout, latest_stats, current_interval, terminal_width)
                else:
                    # Si en pause, reconstruire uniquement la mise en page avec les dernières données
                    if latest_stats:
                        build_layout_content(layout, latest_stats, current_interval, terminal_width=shutil.get_terminal_size(fallback=(80, 20)).columns)

            # Vérifier si un rebuild du layout est requis
            if rebuild_layout_event.is_set():
                if help_flag:
                    help_panel = build_full_screen_help()
                    layout.update(help_panel)
                elif not paused and latest_stats:
                    build_layout_content(layout, latest_stats, current_interval, terminal_width=shutil.get_terminal_size(fallback=(80, 24)).columns)
                elif paused and latest_stats:
                    build_layout_content(layout, latest_stats, current_interval, terminal_width=shutil.get_terminal_size(fallback=(80, 20)).columns)
                rebuild_layout_event.clear()

            # Attendre l'intervalle de rafraîchissement ou un événement
            sleep_time = current_interval
            start_time = time.time()
            while (time.time() - start_time) < sleep_time:
                if exit_event.is_set() or rebuild_layout_event.is_set():
                    break
                time.sleep(0.1)  # Attendre par tranches de 100ms pour réagir rapidement aux événements

    console.print("[bold red]Fermeture du dashboard...[/bold red]")


if __name__ == "__main__":
    main()
