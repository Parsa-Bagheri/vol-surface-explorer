const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const toNumber = (value, fallback) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const setupRangeControl = (control) => {
  const minSlider = control.querySelector(".range-thumb-min");
  const maxSlider = control.querySelector(".range-thumb-max");
  const minNumber = control.querySelector(".range-number:first-of-type");
  const maxNumber = control.querySelector(".range-number:last-of-type");
  const track = control.querySelector(".dual-range");
  const minBound = toNumber(minSlider.min, 0);
  const maxBound = toNumber(minSlider.max, 100);
  const minGap = toNumber(control.dataset.minGap, 0);

  const setValues = (nextMin, nextMax, source) => {
    let minValue = clamp(Math.round(nextMin), minBound, maxBound);
    let maxValue = clamp(Math.round(nextMax), minBound, maxBound);

    if (minValue > maxValue - minGap) {
      if (source === "min") {
        minValue = clamp(maxValue - minGap, minBound, maxBound);
      } else {
        maxValue = clamp(minValue + minGap, minBound, maxBound);
      }
    }

    minSlider.value = String(minValue);
    maxSlider.value = String(maxValue);
    minNumber.value = String(minValue);
    maxNumber.value = String(maxValue);

    const minPercent = ((minValue - minBound) / (maxBound - minBound)) * 100;
    const maxPercent = ((maxValue - minBound) / (maxBound - minBound)) * 100;
    track.style.setProperty("--range-min", `${minPercent}%`);
    track.style.setProperty("--range-max", `${maxPercent}%`);

    minSlider.style.zIndex = minValue > maxBound - 12 ? "3" : "2";
    maxSlider.style.zIndex = "2";
  };

  minSlider.addEventListener("input", () => {
    setValues(toNumber(minSlider.value, minBound), toNumber(maxSlider.value, maxBound), "min");
  });

  maxSlider.addEventListener("input", () => {
    setValues(toNumber(minSlider.value, minBound), toNumber(maxSlider.value, maxBound), "max");
  });

  const syncFromMinNumber = () => {
    if (minNumber.value === "") {
      return;
    }
    setValues(toNumber(minNumber.value, minBound), toNumber(maxNumber.value, maxBound), "min");
  };

  const syncFromMaxNumber = () => {
    if (maxNumber.value === "") {
      return;
    }
    setValues(toNumber(minNumber.value, minBound), toNumber(maxNumber.value, maxBound), "max");
  };

  minNumber.addEventListener("input", syncFromMinNumber);
  minNumber.addEventListener("change", syncFromMinNumber);
  maxNumber.addEventListener("input", syncFromMaxNumber);
  maxNumber.addEventListener("change", syncFromMaxNumber);

  minNumber.addEventListener("focus", () => minNumber.select());
  maxNumber.addEventListener("focus", () => maxNumber.select());

  setValues(toNumber(minNumber.value, minBound), toNumber(maxNumber.value, maxBound), "max");
};

document.querySelectorAll("[data-range-control]").forEach(setupRangeControl);

const setupLoadingState = () => {
  const form = document.querySelector(".control-form");
  if (!form) {
    return;
  }

  const submitButton = form.querySelector(".primary-button");
  const tickerInput = form.querySelector('input[name="ticker"]');
  const loadingStatuses = document.querySelectorAll(
    ".plot-loading-overlay, .plot-loading-placeholder, .loading-surface-card",
  );

  const resetLoading = () => {
    document.body.classList.remove("is-loading", "has-existing-surface");
    form.removeAttribute("aria-busy");
    if (submitButton) {
      submitButton.disabled = false;
    }
    loadingStatuses.forEach((element) => {
      element.setAttribute("aria-hidden", "true");
    });
  };

  form.addEventListener("submit", () => {
    const hasExistingSurface = Boolean(
      document.querySelector(".surface-card:not(.loading-surface-card) .plot-wrapper"),
    );
    const ticker = (tickerInput?.value || "Surface").trim().toUpperCase() || "Surface";

    document.querySelectorAll("[data-loading-ticker]").forEach((element) => {
      element.textContent = ticker;
    });

    document.body.classList.add("is-loading");
    document.body.classList.toggle("has-existing-surface", hasExistingSurface);
    form.setAttribute("aria-busy", "true");
    if (submitButton) {
      submitButton.disabled = true;
    }
    loadingStatuses.forEach((element) => {
      element.setAttribute("aria-hidden", "false");
    });

    if (!hasExistingSurface) {
      document.querySelector(".loading-surface-card")?.scrollIntoView({
        block: "start",
        behavior: "smooth",
      });
    }
  });

  window.addEventListener("pageshow", resetLoading);
};

setupLoadingState();
