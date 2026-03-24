from flask import Flask, request, render_template_string, url_for
from types import SimpleNamespace
from statistics import mean
import os

from federated_cifar10_advanced import federated_train

app = Flask(__name__)

OUTPUT_DIR = os.path.join("static", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Federated Learning Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 16px;
            background: linear-gradient(135deg, #eef4ff, #f8fbff);
            color: #1f2937;
        }

        .container {
            width: 100%;
            max-width: 1100px;
            margin: 0 auto;
            background: #ffffff;
            padding: 20px;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.10);
            border: 1px solid #e5ecf6;
        }

        h1 {
            margin-top: 0;
            font-size: 32px;
            color: #0f172a;
            font-weight: 800;
            line-height: 1.2;
        }

        h2 {
            margin-top: 28px;
            margin-bottom: 14px;
            color: #0f172a;
            font-size: 24px;
        }

        p.note {
            color: #475569;
            font-size: 15px;
            margin-bottom: 24px;
            line-height: 1.6;
        }

        .preset-box {
            background: linear-gradient(135deg, #eff6ff, #e0f2fe);
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 22px;
            color: #1e3a8a;
            font-size: 14px;
            line-height: 1.5;
        }

        .preset-box strong {
            color: #1d4ed8;
        }

        form {
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
        }

        label {
            font-weight: 700;
            display: block;
            margin-bottom: 8px;
            font-size: 15px;
            color: #0f172a;
        }

        input,
        select {
            width: 100%;
            padding: 12px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            font-size: 15px;
            background: #f8fafc;
            color: #1e293b;
            transition: all 0.2s ease;
        }

        input:focus,
        select:focus {
            outline: none;
            border-color: #2563eb;
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.15);
        }

        small {
            display: block;
            margin-top: 6px;
            color: #64748b;
            font-size: 13px;
            line-height: 1.4;
        }

        .full-row {
            grid-column: auto;
        }

        button {
            width: 100%;
            padding: 14px 18px;
            background: linear-gradient(135deg, #2563eb, #0284c7);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 17px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.25s ease;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25);
        }

        button:hover {
            transform: translateY(-1px);
            background: linear-gradient(135deg, #1d4ed8, #0369a1);
            box-shadow: 0 8px 22px rgba(37, 99, 235, 0.30);
        }

        .status {
            margin-top: 24px;
            padding: 15px 18px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 15px;
            line-height: 1.5;
        }

        .success {
            background: #ecfdf3;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        .error {
            background: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 14px;
            margin-top: 22px;
        }

        .summary-card {
            background: linear-gradient(180deg, #ffffff, #f8fbff);
            border: 1px solid #dbe7f3;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        }

        .summary-card p {
            margin: 0 0 8px;
            color: #64748b;
            font-size: 14px;
            font-weight: 600;
        }

        .summary-card h3 {
            margin: 0;
            font-size: 24px;
            color: #2563eb;
            font-weight: 800;
            word-break: break-word;
        }

        .summary-card span {
            display: block;
            margin-top: 6px;
            font-size: 14px;
            color: #475569;
        }

        .images {
            display: grid;
            grid-template-columns: 1fr;
            gap: 18px;
            margin-top: 20px;
        }

        .img-card {
            background: #fbfdff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }

        .img-card h3 {
            margin-top: 0;
            font-size: 17px;
            color: #0f172a;
            line-height: 1.4;
        }

        .img-card img {
            width: 100%;
            height: auto;
            border-radius: 12px;
            border: 1px solid #dbe2ea;
            background: white;
            display: block;
        }

        .table-wrapper {
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin-top: 16px;
            border-radius: 12px;
        }

        table {
            width: 100%;
            min-width: 500px;
            border-collapse: collapse;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
            background: white;
        }

        th,
        td {
            padding: 12px 14px;
            border: 1px solid #e2e8f0;
            text-align: center;
            font-size: 14px;
        }

        th {
            background: linear-gradient(135deg, #dbeafe, #e0f2fe);
            color: #0f172a;
            font-weight: 700;
        }

        tr:nth-child(even) {
            background: #f8fafc;
        }

        tr:hover {
            background: #eef6ff;
        }

        .footer-note {
            margin-top: 22px;
            color: #64748b;
            font-size: 14px;
            text-align: center;
            line-height: 1.5;
        }

        @media (min-width: 640px) {
            body {
                padding: 24px;
            }

            .container {
                padding: 26px;
            }

            h1 {
                font-size: 40px;
            }

            .summary-grid {
                grid-template-columns: 1fr 1fr;
            }

            .images {
                grid-template-columns: 1fr;
            }
        }

        @media (min-width: 900px) {
            body {
                padding: 30px;
            }

            .container {
                padding: 32px;
            }

            h1 {
                font-size: 48px;
            }

            form {
                grid-template-columns: 1fr 1fr;
                gap: 18px 22px;
            }

            .full-row {
                grid-column: span 2;
            }

            .summary-grid {
                grid-template-columns: repeat(4, 1fr);
            }

            .images {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Federated Learning Dashboard</h1>

        <p class="note">
            This dashboard runs <strong>CIFAR-10 federated learning</strong> with non-IID client splits
            and compares <strong>FedAvg</strong> and <strong>FedProx</strong>.
        </p>

        <div class="preset-box">
            <strong>Recommended demo settings:</strong>
            FedAvg, 3 rounds, 4 clients, local epochs = 1, alpha = 0.5.
            These settings run faster and are good for demo and screenshots.
        </div>

        <form method="POST">
            <div>
                <label>Algorithm</label>
                <select name="algorithm">
                    <option value="fedavg" {% if form.algorithm == "fedavg" %}selected{% endif %}>FedAvg</option>
                    <option value="fedprox" {% if form.algorithm == "fedprox" %}selected{% endif %}>FedProx</option>
                </select>
                <small>FedAvg is the standard baseline. FedProx helps when client data is more uneven.</small>
            </div>

            <div>
                <label>Preset</label>
                <select name="preset">
                    <option value="custom" {% if form.preset == "custom" %}selected{% endif %}>Custom</option>
                    <option value="quick" {% if form.preset == "quick" %}selected{% endif %}>Quick Demo</option>
                    <option value="balanced" {% if form.preset == "balanced" %}selected{% endif %}>Balanced</option>
                    <option value="strong" {% if form.preset == "strong" %}selected{% endif %}>Stronger Run</option>
                </select>
                <small>Select a preset or keep custom values.</small>
            </div>

            <div>
                <label>Rounds</label>
                <input type="number" name="rounds" value="{{ form.rounds }}" min="1">
            </div>

            <div>
                <label>Number of Clients</label>
                <input type="number" name="num_clients" value="{{ form.num_clients }}" min="2">
            </div>

            <div>
                <label>Client Fraction</label>
                <input type="number" step="0.1" name="client_fraction" value="{{ form.client_fraction }}">
                <small>Fraction of clients participating in each round.</small>
            </div>

            <div>
                <label>Local Epochs</label>
                <input type="number" name="local_epochs" value="{{ form.local_epochs }}" min="1">
                <small>How long each client trains locally before aggregation.</small>
            </div>

            <div>
                <label>Batch Size</label>
                <input type="number" name="batch_size" value="{{ form.batch_size }}" min="8">
            </div>

            <div>
                <label>Learning Rate</label>
                <input type="number" step="0.0001" name="lr" value="{{ form.lr }}">
            </div>

            <div>
                <label>Alpha (Non-IID Control)</label>
                <input type="number" step="0.1" name="alpha" value="{{ form.alpha }}">
                <small>Smaller alpha means more uneven client data distribution.</small>
            </div>

            <div>
                <label>Mu (FedProx only)</label>
                <input type="number" step="0.001" name="mu" value="{{ form.mu }}">
                <small>Only meaningful when algorithm is FedProx.</small>
            </div>

            <div>
                <label>Seed</label>
                <input type="number" name="seed" value="{{ form.seed }}">
            </div>

            <div class="full-row">
                <button type="submit">🚀 Start Training</button>
                <small style="margin-top:10px;">
                    After clicking, wait for training to finish. The page will reload with graphs and metrics.
                </small>
            </div>
        </form>

        {% if status %}
            <div class="status success">{{ status }}</div>
        {% endif %}

        {% if error %}
            <div class="status error">Error: {{ error }}</div>
        {% endif %}

        {% if summary %}
            <h2>Run Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <p>Average Client Accuracy</p>
                    <h3>{{ "%.4f"|format(summary.avg_acc) }}</h3>
                </div>
                <div class="summary-card">
                    <p>Average Client Loss</p>
                    <h3>{{ "%.4f"|format(summary.avg_loss) }}</h3>
                </div>
                <div class="summary-card">
                    <p>Best Client</p>
                    <h3>Client {{ summary.best_client_id }}</h3>
                    <span>Accuracy: {{ "%.4f"|format(summary.best_client_acc) }}</span>
                </div>
                <div class="summary-card">
                    <p>Lowest Client Accuracy</p>
                    <h3>Client {{ summary.worst_client_id }}</h3>
                    <span>Accuracy: {{ "%.4f"|format(summary.worst_client_acc) }}</span>
                </div>
            </div>
        {% endif %}

        {% if images %}
            <h2>Training Graphs</h2>
            <div class="images">
                {% for img in images %}
                    <div class="img-card">
                        <h3>{{ img.name }}</h3>
                        <img src="{{ img.url }}" alt="{{ img.name }}">
                    </div>
                {% endfor %}
            </div>
        {% endif %}

        {% if client_metrics %}
            <h2>Per-Client Validation Metrics</h2>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Client ID</th>
                            <th>Validation Loss</th>
                            <th>Validation Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for cid, metrics in client_metrics.items() %}
                        <tr>
                            <td>{{ cid }}</td>
                            <td>{{ "%.4f"|format(metrics["val_loss"]) }}</td>
                            <td>{{ "%.4f"|format(metrics["val_acc"]) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% endif %}

        <p class="footer-note">
            Responsive Flask dashboard for federated learning experiments using CIFAR-10, non-IID client splits, FedAvg, and FedProx.
        </p>
    </div>
</body>
</html>
"""

DEFAULT_FORM = {
    "algorithm": "fedavg",
    "preset": "custom",
    "rounds": 5,
    "num_clients": 6,
    "client_fraction": 0.5,
    "local_epochs": 1,
    "batch_size": 64,
    "lr": 0.001,
    "alpha": 0.3,
    "mu": 0.01,
    "seed": 42,
}

PRESETS = {
    "quick": {
        "algorithm": "fedavg",
        "rounds": 3,
        "num_clients": 4,
        "client_fraction": 0.5,
        "local_epochs": 1,
        "batch_size": 64,
        "lr": 0.001,
        "alpha": 0.5,
        "mu": 0.01,
        "seed": 42,
    },
    "balanced": {
        "algorithm": "fedavg",
        "rounds": 5,
        "num_clients": 6,
        "client_fraction": 0.5,
        "local_epochs": 1,
        "batch_size": 64,
        "lr": 0.001,
        "alpha": 0.3,
        "mu": 0.01,
        "seed": 42,
    },
    "strong": {
        "algorithm": "fedprox",
        "rounds": 8,
        "num_clients": 8,
        "client_fraction": 0.6,
        "local_epochs": 2,
        "batch_size": 64,
        "lr": 0.001,
        "alpha": 0.2,
        "mu": 0.02,
        "seed": 42,
    },
}


def build_summary(client_metrics):
    if not client_metrics:
        return None

    accs = [m["val_acc"] for m in client_metrics.values()]
    losses = [m["val_loss"] for m in client_metrics.values()]

    best_client_id, best_client_metrics = max(
        client_metrics.items(), key=lambda item: item[1]["val_acc"]
    )
    worst_client_id, worst_client_metrics = min(
        client_metrics.items(), key=lambda item: item[1]["val_acc"]
    )

    return {
        "avg_acc": mean(accs),
        "avg_loss": mean(losses),
        "best_client_id": best_client_id,
        "best_client_acc": best_client_metrics["val_acc"],
        "worst_client_id": worst_client_id,
        "worst_client_acc": worst_client_metrics["val_acc"],
    }


@app.route("/", methods=["GET", "POST"])
def home():
    form = DEFAULT_FORM.copy()
    status = None
    error = None
    client_metrics = None
    summary = None
    images = []

    if request.method == "POST":
        try:
            preset = request.form.get("preset", "custom")
            form["preset"] = preset

            if preset in PRESETS:
                form.update(PRESETS[preset])
                form["preset"] = preset
            else:
                form["algorithm"] = request.form.get("algorithm", "fedavg")
                form["rounds"] = int(request.form.get("rounds", 5))
                form["num_clients"] = int(request.form.get("num_clients", 6))
                form["client_fraction"] = float(request.form.get("client_fraction", 0.5))
                form["local_epochs"] = int(request.form.get("local_epochs", 1))
                form["batch_size"] = int(request.form.get("batch_size", 64))
                form["lr"] = float(request.form.get("lr", 0.001))
                form["alpha"] = float(request.form.get("alpha", 0.3))
                form["mu"] = float(request.form.get("mu", 0.01))
                form["seed"] = int(request.form.get("seed", 42))

            args = SimpleNamespace(
                algorithm=form["algorithm"],
                rounds=form["rounds"],
                num_clients=form["num_clients"],
                client_fraction=form["client_fraction"],
                local_epochs=form["local_epochs"],
                batch_size=form["batch_size"],
                lr=form["lr"],
                alpha=form["alpha"],
                mu=form["mu"],
                seed=form["seed"],
                data_dir="./data",
                output_dir=OUTPUT_DIR,
            )

            client_metrics = federated_train(args)
            summary = build_summary(client_metrics)
            status = "Training completed successfully."

            algo = form["algorithm"]
            image_names = [
                f"{algo}_train_loss.png",
                f"{algo}_test_loss.png",
                f"{algo}_test_acc.png",
                f"{algo}_client_val_acc.png",
            ]

            images = []
            for img in image_names:
                img_path = os.path.join(OUTPUT_DIR, img)
                if os.path.exists(img_path):
                    images.append({
                        "name": img.replace(".png", "").replace("_", " ").title(),
                        "url": url_for("static", filename=f"outputs/{img}")
                    })

        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        form=form,
        status=status,
        error=error,
        client_metrics=client_metrics,
        summary=summary,
        images=images,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
