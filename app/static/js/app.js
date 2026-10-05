/**
 * Hotel Booking Cancellation Prediction - Frontend Interaction Handler
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('bookingForm');
    const btnPredict = document.getElementById('btnPredict');
    const btnText = btnPredict.querySelector('.btn-text');
    const spinner = btnPredict.querySelector('.spinner');
    const btnReset = document.getElementById('btnReset');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessageText = document.getElementById('errorMessageText');

    const emptyResultContent = document.getElementById('emptyResultContent');
    const activeResultContent = document.getElementById('activeResultContent');
    const riskBadge = document.getElementById('riskBadge');
    const predictionHeadline = document.getElementById('predictionHeadline');
    const probCancelVal = document.getElementById('probCancelVal');
    const probCancelMeter = document.getElementById('probCancelMeter');
    const probNotCancelVal = document.getElementById('probNotCancelVal');
    const probNotCancelMeter = document.getElementById('probNotCancelMeter');

    // Demo Buttons
    const btnDemoLow = document.getElementById('btnDemoLow');
    const btnDemoMod = document.getElementById('btnDemoMod');
    const btnDemoHigh = document.getElementById('btnDemoHigh');

    let demoCasesCache = null;

    // Load Demo Cases from Backend
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

    // Populate Form from Demo Case
    function populateForm(inputs) {
        if (!inputs) return;

        // Clear previous error
        hideError();

        for (const [key, value] of Object.entries(inputs)) {
            const field = form.elements[key];
            if (field) {
                if (value === null || value === undefined) {
                    field.value = '';
                } else {
                    field.value = value;
                }
            }
        }

        // Construct arrival_date if year, month, day are provided
        if (inputs.arrival_date_year && inputs.arrival_date_month && inputs.arrival_date_day_of_month) {
            const monthNames = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"];
            const monthIdx = monthNames.indexOf(inputs.arrival_date_month);
            if (monthIdx >= 0) {
                const yyyy = inputs.arrival_date_year;
                const mm = String(monthIdx + 1).padStart(2, '0');
                const dd = String(inputs.arrival_date_day_of_month).padStart(2, '0');
                const dateField = form.elements['arrival_date'];
                if (dateField) {
                    dateField.value = `${yyyy}-${mm}-${dd}`;
                }
            }
        }

        // Trigger prediction immediately for demo convenience
        submitPrediction();
    }

    // Attach Demo Button Handlers
    btnDemoLow.addEventListener('click', () => {
        if (demoCasesCache && demoCasesCache[0]) {
            populateForm(demoCasesCache[0].raw_booking_inputs);
        } else {
            populateForm({
                hotel: 'City Hotel',
                lead_time: 0,
                arrival_date_year: 2015,
                arrival_date_month: 'October',
                arrival_date_day_of_month: 6,
                stays_in_weekend_nights: 0,
                stays_in_week_nights: 1,
                adults: 2,
                children: 0,
                babies: 0,
                meal: 'BB',
                country: 'PRT',
                market_segment: 'Online TA',
                distribution_channel: 'TA/TO',
                is_repeated_guest: 0,
                previous_cancellations: 0,
                previous_bookings_not_canceled: 0,
                reserved_room_type: 'D',
                assigned_room_type: 'A',
                booking_changes: 0,
                deposit_type: 'No Deposit',
                agent: 9.0,
                adr: 136.0,
                required_car_parking_spaces: 0,
                total_of_special_requests: 1,
                days_in_waiting_list: 0,
                customer_type: 'Contract'
            });
        }
    });

    btnDemoMod.addEventListener('click', () => {
        if (demoCasesCache && demoCasesCache[2]) {
            populateForm(demoCasesCache[2].raw_booking_inputs);
        } else {
            populateForm({
                hotel: 'City Hotel',
                lead_time: 59,
                arrival_date_year: 2017,
                arrival_date_month: 'May',
                arrival_date_day_of_month: 26,
                stays_in_weekend_nights: 1,
                stays_in_week_nights: 2,
                adults: 2,
                children: 0,
                babies: 0,
                meal: 'BB',
                country: 'AUT',
                market_segment: 'Online TA',
                distribution_channel: 'TA/TO',
                is_repeated_guest: 0,
                previous_cancellations: 0,
                previous_bookings_not_canceled: 0,
                reserved_room_type: 'A',
                assigned_room_type: 'A',
                booking_changes: 0,
                deposit_type: 'No Deposit',
                adr: 102.56,
                required_car_parking_spaces: 0,
                total_of_special_requests: 0,
                days_in_waiting_list: 0,
                customer_type: 'Transient'
            });
        }
    });

    btnDemoHigh.addEventListener('click', () => {
        if (demoCasesCache && demoCasesCache[1]) {
            populateForm(demoCasesCache[1].raw_booking_inputs);
        } else {
            populateForm({
                hotel: 'City Hotel',
                lead_time: 187,
                arrival_date_year: 2017,
                arrival_date_month: 'July',
                arrival_date_day_of_month: 17,
                stays_in_weekend_nights: 1,
                stays_in_week_nights: 3,
                adults: 2,
                children: 0,
                babies: 0,
                meal: 'BB',
                country: 'PRT',
                market_segment: 'Online TA',
                distribution_channel: 'TA/TO',
                is_repeated_guest: 0,
                previous_cancellations: 0,
                previous_bookings_not_canceled: 0,
                reserved_room_type: 'D',
                assigned_room_type: 'D',
                booking_changes: 0,
                deposit_type: 'No Deposit',
                adr: 105.3,
                required_car_parking_spaces: 0,
                total_of_special_requests: 0,
                days_in_waiting_list: 0,
                customer_type: 'Transient'
            });
        }
    });

    // Form Submission Handler
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        submitPrediction();
    });

    async function submitPrediction() {
        hideError();
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
                showError(result.message || 'Validation error occurred.');
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

    function renderPrediction(data) {
        emptyResultContent.style.display = 'none';
        activeResultContent.style.display = 'block';

        // Prediction Label
        if (data.prediction === 1) {
            predictionHeadline.textContent = 'LIKELY TO CANCEL';
            predictionHeadline.className = 'prediction-headline text-cancel';
        } else {
            predictionHeadline.textContent = 'LIKELY NOT TO CANCEL';
            predictionHeadline.className = 'prediction-headline text-not-cancel';
        }

        // Risk Badge
        riskBadge.textContent = `${data.risk_level} Risk`;
        riskBadge.className = `risk-badge ${data.risk_badge}`;

        // Cancellation Probability Meter
        const cancelPct = (data.cancellation_probability * 100).toFixed(1);
        probCancelVal.textContent = `${cancelPct}%`;
        probCancelMeter.style.width = `${cancelPct}%`;

        // Non-cancellation Probability Meter
        const notCancelPct = (data.non_cancellation_probability * 100).toFixed(1);
        probNotCancelVal.textContent = `${notCancelPct}%`;
        probNotCancelMeter.style.width = `${notCancelPct}%`;
    }

    function showError(message) {
        errorMessageText.textContent = message;
        errorAlert.style.display = 'flex';
        errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideError() {
        errorAlert.style.display = 'none';
    }

    function setLoading(isLoading) {
        if (isLoading) {
            btnPredict.disabled = true;
            btnText.style.display = 'none';
            spinner.style.display = 'inline-block';
        } else {
            btnPredict.disabled = false;
            btnText.style.display = 'inline-block';
            spinner.style.display = 'none';
        }
    }

    // Reset Form Handler
    btnReset.addEventListener('click', () => {
        form.reset();
        hideError();
        activeResultContent.style.display = 'none';
        emptyResultContent.style.display = 'block';
    });
});
