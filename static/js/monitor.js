document.addEventListener('DOMContentLoaded', () => {
  const refreshBtn = document.getElementById('refreshBtn');
  refreshBtn?.addEventListener('click', () => location.reload());

  // Auto-refresh every 60 seconds to help keep monitor page up-to-date
  const AUTO_REFRESH_MS = 60000;
  let timer = setInterval(() => location.reload(), AUTO_REFRESH_MS);

  // Pause auto-refresh while the user hovers the table
  const table = document.getElementById('checkpointsTable');
  table?.addEventListener('mouseenter', () => clearInterval(timer));
  table?.addEventListener('mouseleave', () => {
    timer = setInterval(() => location.reload(), AUTO_REFRESH_MS);
  });
});
