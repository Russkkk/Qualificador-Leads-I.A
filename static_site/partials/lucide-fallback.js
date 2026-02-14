(function () {
  if (window.lucide) return;
  var s = document.createElement("script");
  s.src = "https://cdn.jsdelivr.net/npm/lucide@latest/dist/umd/lucide.min.js";
  s.defer = true;
  s.onload = function () {
    try {
      window.lucide && window.lucide.createIcons();
    } catch (_) {}
  };
  document.head.appendChild(s);
})();
