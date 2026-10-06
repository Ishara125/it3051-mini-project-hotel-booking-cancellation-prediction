/**
 * Hotel Booking Cancellation Prediction — Predictor Page Interactions
 * IT3051 Fundamentals of Data Mining
 */

document.addEventListener('DOMContentLoaded', () => {
  /* ---- Element references ---- */
  const form          = document.getElementById('bookingForm');
  const btnPredict    = document.getElementById('btnPredict');
  const btnText       = btnPredict.querySelector('.btn-text');
  const btnLoading    = btnPredict.querySelector('.btn-loading');
  const btnReset      = document.getElementById('btnReset');
  const errorAlert    = document.getElementById('errorAlert');
  const errorMsgText  = document.getElementById('errorMessageText');

  const emptyState    = document.getElementById('emptyState');
  const activeResult  = document.getElementById('activeResult');
  const riskBanner    = document.getElementById('riskBanner');
  const riskLevel     = document.getElementById('riskLevel');
  const riskVerdict   = document.getElementById('riskVerdict');
  const probCancelVal = document.getElementById('probCancelVal');
  const probCancelMeter    = document.getElementById('probCancelMeter');
  const probCancelTrack    = document.getElementById('probCancelTrack');
  const probNotCancelVal   = document.getElementById('probNotCancelVal');
  const probNotCancelMeter = document.getElementById('probNotCancelMeter');
  const probNotCancelTrack = document.getElementById('probNotCancelTrack');
  const resultContext = document.getElementById('resultContext');

  const btnDemoLow  = document.getElementById('btnDemoLow');
  const btnDemoMod  = document.getElementById('btnDemoMod');
  const btnDemoHigh = document.getElementById('btnDemoHigh');

  let demoCasesCache = null;

  /* ---- Load demo cases from backend ---- */
  async function loadDemoCases() {
    try {
      const resp = await fetch('/api/demo-cases');
      if (resp.ok) {
        demoCasesCache = await resp.json();
      }
    } catch (err) {
      console.warn('Could not pre-fetch demo cases:', err);
    }
  }
  loadDemoCases();

  /* ---- Populate form from a demo case object ---- */
  function populateForm(inputs) {
    if (!inputs) return;
    hideError();
    clearFieldErrors();

    for (const [key, value] of Object.entries(inputs)) {
      const field = form.elements[key];
      if (field) {
        field.value = (value === null || value === undefined) ? '' : value;
      }
    }

    // Construct arrival_date from year/month/day if provided
    if (inputs.arrival_date_year && inputs.arrival_date_month && inputs.arrival_date_day_of_month) {
      const monthNames = [
        'January','February','March','April','May','June',
        'July','August','September','October','November','December'
      ];
      const monthIdx = monthNames.indexOf(inputs.arrival_date_month);
      if (monthIdx >= 0) {
        const yyyy = inputs.arrival_date_year;
        const mm   = String(monthIdx + 1).padStart(2, '0');
        const dd   = String(inputs.arrival_date_day_of_month).padStart(2, '0');
        const dateField = form.elements['arrival_date'];
        if (dateField) dateField.value = `${yyyy}-${mm}-${dd}`;
      }
    }

    // Auto-submit for demo convenience
    submitPrediction();
  }

  /* ---- Demo button handlers ---- */
  btnDemoLow.addEventListener('click', () => {
    if (demoCasesCache && demoCasesCache[0]) {
      populateForm(demoCasesCache[0].raw_booking_inputs);
    } else {
      populateForm({
        hotel: 'City Hotel', lead_time: 0,
        arrival_date_year: 2015, arrival_date_month: 'October', arrival_date_day_of_month: 6,
        stays_in_weekend_nights: 0, stays_in_week_nights: 1,
        adults: 2, children: 0, babies: 0, meal: 'BB', country: 'PRT',
        market_segment: 'Online TA', distribution_channel: 'TA/TO',
        is_repeated_guest: 0, previous_cancellations: 0, previous_bookings_not_canceled: 0,
        reserved_room_type: 'D', assigned_room_type: 'A',
        booking_changes: 0, deposit_type: 'No Deposit',
        agent: 9.0, adr: 136.0,
        required_car_parking_spaces: 0, total_of_special_requests: 1,
        days_in_waiting_list: 0, customer_type: 'Contract'
      });
    }
  });

  btnDemoMod.addEventListener('click', () => {
    if (demoCasesCache && demoCasesCache[2]) {
      populateForm(demoCasesCache[2].raw_booking_inputs);
    } else {
      populateForm({
        hotel: 'City Hotel', lead_time: 59,
        arrival_date_year: 2017, arrival_date_month: 'May', arrival_date_day_of_month: 26,
        stays_in_weekend_nights: 1, stays_in_week_nights: 2,
        adults: 2, children: 0, babies: 0, meal: 'BB', country: 'AUT',
        market_segment: 'Online TA', distribution_channel: 'TA/TO',
        is_repeated_guest: 0, previous_cancellations: 0, previous_bookings_not_canceled: 0,
        reserved_room_type: 'A', assigned_room_type: 'A',
        booking_changes: 0, deposit_type: 'No Deposit',
        adr: 102.56,
        required_car_parking_spaces: 0, total_of_special_requests: 0,
        days_in_waiting_list: 0, customer_type: 'Transient'
      });
    }
  });

  btnDemoHigh.addEventListener('click', () => {
    if (demoCasesCache && demoCasesCache[1]) {
      populateForm(demoCasesCache[1].raw_booking_inputs);
    } else {
      populateForm({
        hotel: 'City Hotel', lead_time: 187,
        arrival_date_year: 2017, arrival_date_month: 'July', arrival_date_day_of_month: 17,
        stays_in_weekend_nights: 1, stays_in_week_nights: 3,
        adults: 2, children: 0, babies: 0, meal: 'BB', country: 'PRT',
        market_segment: 'Online TA', distribution_channel: 'TA/TO',
        is_repeated_guest: 0, previous_cancellations: 0, previous_bookings_not_canceled: 0,
        reserved_room_type: 'D', assigned_room_type: 'D',
        booking_changes: 0, deposit_type: 'No Deposit',
        adr: 105.3,
        required_car_parking_spaces: 0, total_of_special_requests: 0,
        days_in_waiting_list: 0, customer_type: 'Transient'
      });
    }
  });

  /* ---- Form submit ---- */
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    submitPrediction();
  });

  async function submitPrediction() {
    hideError();
    clearFieldErrors();
    setLoading(true);

    const formData = new FormData(form);
    const payload = {};
    for (const [key, value] of formData.entries()) {
      payload[key] = value.trim();
    }

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const result = await response.json();

      if (!response.ok) {
        handleError(result);
        return;
      }

      renderPrediction(result.data);

    } catch (err) {
      console.error('Prediction request failed:', err);
      showError('Unable to connect to the prediction server. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  }

  /* ---- Render prediction result ---- */
  function renderPrediction(data) {
    emptyState.style.display = 'none';
    activeResult.style.display = 'block';

    const cancelPct    = (data.cancellation_probability * 100).toFixed(1);
    const notCancelPct = (data.non_cancellation_probability * 100).toFixed(1);

    // Risk banner
    const riskClass = data.risk_badge || 'risk-low';
    riskBanner.className = `risk-banner ${riskClass}`;
    riskLevel.textContent = `${(data.risk_level || 'LOW').toUpperCase()} RISK`;

    if (data.prediction === 1) {
      riskVerdict.textContent = 'Likely to Cancel';
    } else {
      riskVerdict.textContent = 'Likely Not to Cancel';
    }

    // Probabilities
    probCancelVal.textContent = `${cancelPct}%`;
    probCancelMeter.style.width = `${cancelPct}%`;
    probCancelTrack.setAttribute('aria-valuenow', cancelPct);

    probNotCancelVal.textContent = `${notCancelPct}%`;
    probNotCancelMeter.style.width = `${notCancelPct}%`;
    probNotCancelTrack.setAttribute('aria-valuenow', notCancelPct);

    // Contextual message
    resultContext.textContent = getContextMessage(data.prediction, parseFloat(cancelPct));

    // Scroll result panel into view on mobile
    if (window.innerWidth < 1024) {
      setTimeout(() => {
        document.querySelector('.result-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 100);
    }
  }

  function getContextMessage(prediction, cancelPct) {
    if (prediction === 1) {
      if (cancelPct >= 70) {
        return 'This booking has a relatively high predicted cancellation probability based on patterns learned from historical reservations. Consider reviewing the booking with additional attention.';
      }
      return 'This booking is predicted to be at elevated cancellation risk based on patterns learned from historical data. This is an estimate and should inform, not determine, decision-making.';
    } else {
      if (cancelPct < 20) {
        return 'This booking has a relatively low predicted cancellation probability based on patterns learned from historical reservations.';
      }
      return 'This booking is predicted to likely be honoured, though some cancellation probability remains. This estimate is based on historical booking patterns.';
    }
  }

  /* ---- Error handling ---- */
  function handleError(result) {
    const message = result.message || 'An error occurred. Please check your inputs.';
    const errors  = result.errors  || {};

    showError(message);

    // Highlight specific fields if the backend identifies them
    Object.keys(errors).forEach(fieldName => {
      const field = form.elements[fieldName];
      if (field) {
        const group = field.closest('.form-group');
        if (group) {
          group.classList.add('has-error');
          const errEl = group.querySelector('.field-error');
          if (errEl) {
            errEl.textContent = errors[fieldName];
            errEl.style.display = 'block';
          }
        }
      }
    });
  }

  function showError(message) {
    errorMsgText.textContent = message;
    errorAlert.classList.add('visible');
    errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideError() {
    errorAlert.classList.remove('visible');
  }

  function clearFieldErrors() {
    form.querySelectorAll('.form-group.has-error').forEach(group => {
      group.classList.remove('has-error');
      const errEl = group.querySelector('.field-error');
      if (errEl) { errEl.textContent = ''; errEl.style.display = 'none'; }
    });
  }

  /* ---- Loading state ---- */
  function setLoading(isLoading) {
    btnPredict.disabled = isLoading;
    if (isLoading) {
      btnText.style.display = 'none';
      btnLoading.style.display = 'inline-flex';
    } else {
      btnText.style.display = 'inline-flex';
      btnLoading.style.display = 'none';
    }
  }

  /* ---- Reset ---- */
  btnReset.addEventListener('click', () => {
    form.reset();
    hideError();
    clearFieldErrors();
    activeResult.style.display = 'none';
    emptyState.style.display = 'block';
  });

});
