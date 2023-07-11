function render_eventlog(id, data) {
    let csv = [];
    for(let i = 0; i < data.length; i++){

		csv.push({
			'type': data[i].type,
			'time': data[i].time,
			'job': data[i].job,
			'op': data[i].op,
            'm': data[i].m,
            'next': data[i].next
		});
    }

	// column definitions
    let columns = [
        { head: 'Event Type', cl: 'type', html: function(r) { return r.type; } },
        { head: 'Occurrence Time', cl: 'num', html: function(r) { return r.time; } },
        { head: 'Job Index', cl: 'center', html: function(r) { return r.job; } },
        { head: 'Operation Index', cl: 'center', html: function(r) { return r.op; } },
        { head: 'Machine Index', cl: 'center', html: function(r) { return r.m; } },
        { head: 'Triggers Next', cl: 'center', html: function(r) { return r.next; } }
    ];

    // create table
    let table = d3.select('#' + id)
        .append('table').attr('class', "table table-striped");

    // create table header
    table.append('thead').append('tr')
        .selectAll('th')
        .data(columns).enter()
        .append('th')
        .attr('class', function(r) { return r.cl; })
        .text(function(r) { return r.head; });

    // create table body
    table.append('tbody')
        .selectAll('tr')
        .data(csv).enter()
        .append('tr')
        .selectAll('td')
        .data(function(row, i) {
            return columns.map(function(c) {
                // compute cell values for this specific row
                let cell = {};
                Object.keys(c).forEach(function(k) {
                    cell[k] = typeof c[k] == 'function' ? c[k](row,i) : c[k];
                });
                return cell;
            });
        }).enter()
        .append('td')
        .html(function(r) { return r.html; })
        .attr('class', function(r) { return r.cl; });
}