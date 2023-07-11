let initialpoll;
get_run_list();

function toggle_setting_dropdown(el) {
    if (el.parentElement.classList.contains('visible')) {
        el.parentElement.classList.remove('visible');
    }
    else {
        el.parentElement.classList.add('visible');
    }
}

function toggle_matrix_visibililty(checkbox) {
    if (checkbox.checked) {
        let matrix_container = $('#' + checkbox.id + '_view');
        if (matrix_container.length !== 0) {
            matrix_container.show()
        } else {
            let current_id = $('#current_id').text();
            let wip_only = $('#all_jobs').text() !== "1";
            $.get({
                url: '/state_data/' + current_id,
                success: function(data){
                    render_heatmap(
                        data['matrices'][checkbox.id], checkbox.id,
                        data['management_info'],
                        'gr' + current_id + '_matrices',
                        wip_only);
                },
                async: false
           });
        }
    } else {
        $('#' + checkbox.id + '_view').hide()
    }
}

function toggle_graph_visibililty(checkbox) {
    if (checkbox.checked) {
        let matrix_container = $('#' + checkbox.id + '_view');
        if (matrix_container.length !== 0) {
            matrix_container.show()
        } else {
            let current_id = $('#current_id').text();
            $.get({
                url: '/state_data/' + current_id,
                success: function(data){
                    render_graph('gr' + current_id + '_precedence',
                        data['precedence_graphs'][checkbox.id],
                        split_capitalize(checkbox.id, ' ') + 'Precedence Constraints',
                        checkbox.id + '_view');
                },
                async: false
           });
        }
    } else {
        $('#' + checkbox.id + '_view').hide()
    }
}

function toggle_schedule_visibililty(checkbox) {
    let view_name = checkbox.id;
    if (checkbox.checked) {
        let schedule_container = $('#' + view_name + '_view');
        if (schedule_container.length !== 0) {
            schedule_container.show()
        } else {
            let current_id = $('#current_id').text();
            $.get({
                url: '/state_data/' + current_id,
                success: function(data){
                    if (data.hasOwnProperty(view_name)) {
                        drawgantt('gr' + current_id + '_schedule',
                        data[view_name], view_name + '_view',
                        split_capitalize(view_name));
                    }
                },
                async: false
           });
        }
    } else {
        $('#' + view_name + '_view').hide()
    }
}

function toggle_kpi_visibililty(checkbox) {
    let view_name = checkbox.id; // 'metrics_jobs', 'metrics_machines'
    if (checkbox.checked) {
        let kpi_container = $('#' + view_name + '_view');
        if (kpi_container.length !== 0) {
            kpi_container.show()
        } else {
            let current_id = $('#current_id').text();
            $.get({
                url: '/state_data/' + current_id,
                success: function(data){
                    if (data.hasOwnProperty(view_name)){
                        drawKPI('gr' + current_id + '_kpi',
                            view_name + '_view',
                            data[view_name][0],
                            data[view_name][1]);
                    }
                },
                async: false
           });
        }
    } else {
        $('#' + view_name + '_view').hide()
    }
}

function poll(latest_state) {
    /**
     * Asynchronously checks whether the backend received a new state and
     * updates the view if required.
     *
     * If the call available at the /update_required/<state_id> urls returns
     * true, the update_page function is called. Thereafter the last rendered
     * state id is incremented and the polling continues. Should the backend
     * call return false, the polling resumes every 10 seconds without
     * incrementing the last rendered id.
     *
     * @param {int} id  The id of the last rendered production scheduling state.
     * @see update_page
     */
    let defslide = 0;

    $.get({
        url: '/update_required/' + latest_state,
        success: function (response) {
            if (response[0] === true) {
                let state_container_id = "gr"+ latest_state;
                if (latest_state === 0) {
                    $('<div id=' + state_container_id + '>' +
                    '</div>').appendTo('div#vis_root');
                    update_page(0);
                } else {
                    $('<div style="display: none" id=' + state_container_id + '>' +
                    '</div>').appendTo('div#vis_root');
                }
                drawSlider(latest_state, defslide);
                latest_state += 1;
                poll(latest_state);
            } else {
                initialpoll = setTimeout(poll.bind(null, latest_state),
                    10000)
            }
        },
        async: true
    });
}

function get_run_list() {
    $.get({
        url: '/get_run_list',
        success: function (response) {
            for (let i=0; i < response.length; i++) {
                let style_string = i === 0? 'style="color: #0094ff;"' : '';
                 $('<li class="dropdown-item runs-item">' +
                     '<span id="run' + i +'" onclick="switch_run(this)" ' + style_string +
                     '>' + response[i] +
                     '</span></li>'
                 ).appendTo('#runs_dropdown');
            }
        },
        async: true
    });
    let runs_container = $('#runs');
    runs_container.trigger('click');
    poll(0)
}

function switch_run(run_li){
    let parent = $(run_li).closest('.list-group');
    let brothers = parent.find('span');
    brothers.css('color', "#000000");
    let run_container = $(run_li).text();
    $(run_li).css('color', "#0094ff");
    $.get({ //TODO: do this with a post...
        url: '/set_logdir/' + run_container,
        success: function (response) {
            // disable polling first, then poll again ;)
            $('#vis_root').empty();
            clearTimeout(initialpoll);
            poll(0);
        },
        async: true
    });
}

function toggle_run_menu() {
    $('#runs').find( "ul").toggle()
}

function toggle_visible_matrix_jobs(radio_label) {
    let radio_btn = $(radio_label).children()[0];
    let all_jobs = radio_btn.value;
    let current_state_id = $('#current_id').text();
    let state_toggled = 'gr' + current_state_id;
    // call backend and tell it not to send unknown jobs
    // $.get({
    //     url: '/set_matrix_visibility/' +  all_jobs,
    //     success: function (data) {
    //
    //     },
    //     async: true
    // });
    // remove the old data
    $('#' + state_toggled + '_matrices').empty();
    // call backend to get data
    let wip_only = all_jobs !== "1";
    $('#all_jobs').text(all_jobs);
    $.get({
        url: '/state_data/' +  current_state_id,
        success: function (data) {
            // call usual matrix drawing function with reduce data ;)
            render_selected_heatmaps(
                        data['matrices'],
                        data['management_info'],
                        state_toggled + '_matrices', wip_only)
            },
        async: true
    });
}

function update_page(id) {
    /**
     * Synchronously calls the backend function "prepare_state_data" over the
     * "[base_address]/state_data/[state_id]" URL to obtain the state json to be
     * rendered in its dedicated DOM container. The local function
     * "render_state" is then used to draw the different state plots.
     *
     * @param {int} id The identifier of the state to render.
     */
   $.get({
        url: '/state_data/' + id,
        success: function(data){render_state(id, data)},
        async: false
   });
}

function render_state(id, data) {
    /**
    * Uses the drawing functions to render state components based on the
    * json data returned by the backend.
    *
    * @param {object} data The json object with the rendering information.
    */
    $('p#sys_time').text('Time: ' + data['management_info']['system_time']);
    let state_segments = [
        'precedence', 'machines', 'matrices',
        'schedule', 'kpis', 'events'];
    let container_ids = create_dom_containers(id, state_segments);
    if (data.hasOwnProperty('precedence_graphs')){
        if (document.getElementById('visible').checked)
            render_graph(container_ids['precedence'],
                data.precedence_graphs['visible'],
                "Visible Precedence Constraints",
                "visible_view");
        if (document.getElementById('hidden').checked)
            render_graph(container_ids['precedence'],
                data.precedence_graphs['hidden'],
                "Hidden Precedence Constraints",
                "hidden_view");
    }
    if (data.hasOwnProperty('machines'))
        render_machines(container_ids['machines'], data.machines);
    if (data.hasOwnProperty('matrices') &&
        data.hasOwnProperty('management_info')) {
        let wip_only = $('#all_jobs').text() !== "1";
        render_selected_heatmaps(data.matrices, data.management_info,
            container_ids['matrices'], wip_only)
    }
    if (document.getElementById('partial_schedule').checked){
        if (data.hasOwnProperty('partial_schedule'))
            drawgantt(container_ids['schedule'], data.partial_schedule,
                'partial_schedule_view', 'Partial Schedule');
    }
    if (document.getElementById('current_plan').checked) {
        if (data.hasOwnProperty('current_plan')) {
            drawgantt(container_ids['schedule'], data.current_plan,
            'Current Plan');
        }
    }
    if (document.getElementById('metrics_jobs').checked) {
        if (data.hasOwnProperty('metrics_jobs')) {
            drawKPI(container_ids['kpis'], 'metrics_jobs_view',
                data.metrics_jobs[0], data.metrics_jobs[1]);
        }
    }
    if (document.getElementById('metrics_machines').checked) {
        if (data.hasOwnProperty('metrics_machines')) {
            drawKPI(container_ids['kpis'], 'metrics_machines_view',
                data.metrics_machines[0], data.metrics_machines[1]);
        }
    }
    if (data.hasOwnProperty('pending_events'))
        render_eventlog(container_ids['events'], data.pending_events)
}

function create_dom_containers(id, state_segments) {
    let state_container_id = 'gr' + id;
    let container_ids = {};
    for (let i=0; i < state_segments.length; i++) {
        container_ids[state_segments[i]] = add_container(
            state_segments[i])
    }
    function add_container(container_postfix) {
        let state_segment_id = state_container_id + '_' + container_postfix;
        $('<span id=' + state_segment_id + '></span>').appendTo(
            'div#' + state_container_id);
        return state_segment_id
    }
    return container_ids
}

function render_selected_heatmaps(data, markup, container_id, wip_only=false) {
    let draw_later_1 = [];
    let draw_later_2 = [];
    // iterate over obj properties
    for (const matrix_name in data) {
        if (data.hasOwnProperty(matrix_name)) {
            if (document.getElementById(matrix_name).checked) {
                let matrix_object = data[matrix_name];
                if (matrix_object['nfo_type'] === 'jobs')
                    render_heatmap(
                        matrix_object, matrix_name, markup, container_id,
                        wip_only);
                else if (matrix_object['nfo_type'] === 'tracking'){
                    draw_later_1.push([matrix_object, matrix_name]);
                } else {
                    draw_later_2.push([matrix_object, matrix_name])
                }
            }
        }
    }
    for (const m_info of draw_later_1){
       render_heatmap(m_info[0], m_info[1], markup, container_id, wip_only);
    }
    for (const m_info of draw_later_2){
       render_heatmap(m_info[0], m_info[1], markup, container_id, wip_only);
    }
}