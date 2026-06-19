window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  // Navbar burger toggle
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // Qualitative results carousel(s)
  bulmaCarousel.attach('.carousel', {
    slidesToScroll: 1,
    slidesToShow: 5,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000,
  });

  // Bulma sliders
  bulmaSlider.attach();

  // ---- Interactive mixer: Original Video <-> Slot Map ----
  // Each dataset group has its own slider that sets the opacity of the top
  // "slot map" layers within that group only, reproducing
  // image * (1 - a) + slotmap * a live in the browser.
  document.querySelectorAll('.mix-group').forEach(function (group) {
    var slider = group.querySelector('.mix-slider');
    var label = group.querySelector('.mix-value');
    var slots = group.querySelectorAll('.mix-slot');
    function applyMix(v) {
      var alpha = v / 100;
      slots.forEach(function (el) { el.style.opacity = alpha; });
      if (label) label.textContent = Math.round(v) + '%';
    }
    if (slider) {
      slider.addEventListener('input', function (e) { applyMix(e.target.value); });
      applyMix(slider.value);
    }
  });

  // Keep each (base video, slot-map video) pair roughly in sync.
  document.querySelectorAll('.mix-card').forEach(function (card) {
    var base = card.querySelector('.mix-base');
    var slot = card.querySelector('.mix-slot');
    if (base && slot) {
      base.addEventListener('timeupdate', function () {
        if (Math.abs(slot.currentTime - base.currentTime) > 0.08) {
          slot.currentTime = base.currentTime;
        }
      });
    }
  });

  // ---- Slow playback for YouTube-VIS clips (0.5x = 2x slower) ----
  // playbackRate cannot be set via HTML attribute and may reset on (re)load,
  // so apply it now and again once metadata is ready.
  function applySlow(v) {
    var s = v.querySelector('source');
    if (s && s.src.indexOf('/ytvis2021/') !== -1) { v.playbackRate = 0.5; }
  }
  document.querySelectorAll('video').forEach(function (v) {
    applySlow(v);
    v.addEventListener('loadedmetadata', function () { applySlow(v); });
  });

  // ---- Performance: only play videos that are on screen ----
  // The page hosts many looping clips; pausing off-screen ones keeps it smooth.
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting) { applySlow(v); var p = v.play(); if (p && p.catch) p.catch(function () {}); }
        else { v.pause(); }
      });
    }, { threshold: 0.1 });
    document.querySelectorAll('video').forEach(function (v) { io.observe(v); });
  }
});
