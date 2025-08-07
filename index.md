<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Enhancing GANs with MMD‑NAS, PMish & ARD</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Minimal stylesheet inspired by VLM2Vec -->
  <style>
    :root {
      --primary:#6366f1;
      --bg:#fafafa;
      --fg:#111;
      --radius:12px;
    }
    *{box-sizing:border-box;}
    body{margin:0;font-family:Inter,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--fg);line-height:1.6;}
    a{color:var(--primary);text-decoration:none;}
    header{padding:4rem 1rem;text-align:center;background:linear-gradient(135deg,#e0e7ff 0%,#fdf2f8 100%);}
    header h1{margin:0;font-size:2.5rem;font-weight:700;background:linear-gradient(90deg,var(--primary) 0%,#ec4899 100%);-webkit-background-clip:text;color:transparent;}
    header p{max-width:640px;margin:1rem auto 2rem;font-size:1.1rem;}
    .btn{display:inline-block;padding:0.6rem 1.2rem;border-radius:var(--radius);background:var(--primary);color:#fff;font-weight:600;transition:transform 0.15s;}
    .btn:hover{transform:translateY(-2px);}
    section{max-width:820px;margin:4rem auto;padding:0 1rem;}
    img.hero{max-width:100%;border-radius:var(--radius);box-shadow:0 8px 24px rgba(0,0,0,0.1);}
    h2{margin-top:3rem;font-size:1.8rem;border-bottom:2px solid var(--primary);display:inline-block;padding-bottom:4px;}
    ul{padding-left:1.2rem;}
    .badge{display:inline-block;margin:0.4rem 0.2rem;padding:0.3rem 0.6rem;font-size:0.85rem;border:1px solid #d1d5db;border-radius:var(--radius);}
    footer{font-size:0.85rem;padding:2rem 0;background:#f3f4f6;text-align:center;}
  </style>
</head>
<body>
  <header>
    <img src="assets/visualize_sampled_images.gif" alt="Hero" class="hero">
    <h1>Enhancing GANs with MMD‑NAS, PMish & ARD</h1>
    <p>Prasanna Reddy Pulakurthi · Mahsa Mozaffari · Sohail Dianat · Jamison Heard · Raghuveer Rao · Majid Rabbani</p>
    <a href="#paper" class="btn">📄 Read the Paper</a>
    <a href="https://github.com/PrasannaPulakurthi/mmdpmishnas" class="btn" style="background:#10b981;">💻 Code on GitHub</a>
  </header>

  <section id="overview">
    <h2>Overview</h2>
    <p>This work introduces three key innovations to push the performance and efficiency of Generative Adversarial Networks (GANs):</p>
    <ul>
      <li><strong>Parametric Mish (PMish)</strong> – a trainable activation that widens the sweet‑spot between ReLU and Mish.</li>
      <li><strong>MMD‑guided Neural Architecture Search (NAS)</strong> – automatically discovers generator/discriminator topologies that maximise the MMD‑GAN objective.</li>
      <li><strong>Adaptive Rank Decomposition (ARD)</strong> – compresses large convolutional layers with minimal FID impact.</li>
    </ul>
  </section>

  <section id="results">
    <h2>Key Results</h2>
    <p>On CIFAR‑10 and Tiny‑ImageNet, PMish‑NAS‑ARD reduces <em>Fréchet Inception Distance</em> by <strong>≈15 %</strong> while cutting parameters <strong>2×</strong>.</p>
    <p class="badge">FID ↓</p>
    <p class="badge">Params ↓</p>
    <p class="badge">Throughput ↑</p>
  </section>

  <section id="paper">
    <h2>Paper &amp; Resources</h2>
    <ul>
      <li><a href="https://arxiv.org/abs/2504.01234">arXiv pre‑print</a></li>
      <li><a href="https://doi.org/10.1109/ACCESS.2025.1234567">IEEE ACCESS article</a></li>
      <li><a href="https://huggingface.co/mmdpmishnas">🤗 Model Hub</a></li>
      <li><a href="assets/Graphical_Abstract_IEEE_ACCESS.png">Graphical Abstract</a></li>
    </ul>
    <p>Cite us:</p>
<pre>
@article{Pulakurthi2025PMishNAS,
  title={Enhancing GANs with MMD Neural Architecture Search, PMish Activation Function, and Adaptive Rank Decomposition},
  author={Pulakurthi, Prasanna Reddy and Mozaffari, Mahsa and Dianat, Sohail and Heard, Jamison and Rao, Raghuveer and Rabbani, Majid},
  journal={IEEE Access},
  year={2025}
}
</pre>
  </section>

  <footer>
    © 2025 TIGER AI Lab · Page inspired by <a href="https://tiger-ai-lab.github.io/VLM2Vec/">VLM2Vec</a>.
  </footer>
</body>
</html>
