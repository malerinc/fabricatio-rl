function drawgantt(id, taskData, svg_id="Partial Schedule",
                   title="Schedule (Job Start Times)"){
    let tasks = [];
    try{
        for(let i = 0; i < taskData.length; i++){
        tasks.push({taskName: taskData[i][0],
                   startDate: taskData[i][1],
                   endDate: taskData[i][2],
                   status: taskData[i][3]})
        }
    }
    catch(err) {

    }
    function onlyUnique(value, index, self) {
		  return self.indexOf(value) === index;
		}
	let taskNames = tasks.map(a => a.taskName).filter(onlyUnique);
    let taskStatus = tasks.map(a => a.status).filter(onlyUnique);
    // maxtime assignment
    let maxTime = Math.max.apply(Math, tasks.map(function(o) {
        return o.endDate;
    }));
    let rem = maxTime % 200;
    let timeBound = maxTime - rem + 200;
    tasks.sort(function(a, b) {
        return a.endDate - b.endDate;
    });
    tasks.sort(function(a, b) {
        return a.startDate - b.startDate;
    });
    let format = "%s";
    let gantt = d3.gantt()
        .taskTypes(taskNames)
        .taskStatus(taskStatus)
        .tickFormat(format)
        .timeBound(timeBound);
    return gantt(tasks, id, svg_id, title);
}

d3.gantt = function() {
    let margin = {
        top : 40,
        right : 200,
        bottom : 10,
        left : 30
    };
    let timeDomainStart = 0;
    let timeDomainEnd = 4000;
    let timeDomainMode = "fixed";
    let taskTypes = [];
    let taskStatus = [];
    let height = 400 - margin.top - margin.bottom-5;
    let width_abs = 600;
    let width = width_abs - margin.right - margin.left-5;
    let tickFormat = "%s";
    let keyFunction = function(d) {
    return d.startDate + d.taskName + d.endDate;
    };
    let rectTransform = function(d) {
    return "translate(" + x(d.startDate) + "," + y(d.taskName) + ")";
    };
    let x,y,xAxis,yAxis;
    initAxis();


    function initAxis() {
        x = d3.scaleTime().domain([ timeDomainStart, timeDomainEnd ])
            .range([ 0, width + 50]).clamp(true);
        y = d3.scaleBand().domain(taskTypes)
            .rangeRound([ 0, height - margin.top - margin.bottom ], .1);
        xAxis = d3.axisBottom().scale(x).tickFormat(d3.format("d"))
          .tickSize(8).tickPadding(8);
        yAxis = d3.axisLeft().scale(y).tickSize(0);
    }

    function gantt(tasks, id, svg_id, title) {
        initAxis();
        let svg = d3.select('#' + id)
            .append("svg").attr("id", svg_id)
            .attr("class", "zoom")
            .attr("class", "chart")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("class", "gantt-chart")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .attr("transform",
              "translate(" + margin.left + ", " + margin.top + ")");
        // Box
        svg.append('rect')
            .attr('x', -25)
            .attr('y', -10)
            .attr('width', width_abs)
            .attr('height', 360)
            .attr('stroke', 'black')
            .attr('fill', 'white');
        // Figure name
        svg.append("text")
            .attr("x", (width / 2) + 20)
            .attr("y", 340)
            .attr("text-anchor", "middle")
            .style("font-size", "14px")
            .style("font-weight", 700)
            .style("text-decoration", "underline")
            .text(title);

        svg.append("g")
            .attr("width", width)
            .attr("height", height);

        svg.append("rect")
            .attr("width", "85%")
            .attr("height", "75%")
            .attr("fill", "#c1d7e7");

        svg.append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        // gridlines in y axis function
        function make_y_gridlines() {
            return d3.axisLeft(y)
                .ticks(5)
        }
        // add the Y gridlines
        svg.append("g")
          .attr("class", "grid")
          .call(make_y_gridlines()
          .tickSize(-width-50)
          .tickFormat(""));

        // create a list of keys
        let keys = tasks.map(a => a.status).filter(onlyUnique).sort(
            function(a, b) { return a - b});

        let taskNames = tasks.map(a => a.taskName).filter(onlyUnique);
        let keylen = 300 / (taskNames.length) / 2 - 12;

        let colorList = ["#3957ff", "#c9080a", "#0b7b3e", "#0bf0e9", "#fd9b39",
            "#888593", "#906407", "#98ba7f", "#fe6794", "#10b0ff", "#964c63",
            "#1da49c", "#0ad811", "#bbd9fd", "#fe6cfe", "#297192", "#d1a09c",
            "#78579e", "#81ffad", "#739400", "#ca6949"];

        svg.selectAll(".chart")
            .data(tasks, keyFunction).enter()
            .append("rect")
            .attr("rx", 5)
            .attr("ry", 5)
            .attr("fill", function(d) { return colorList[d.status]; })
            .attr("y", keylen)
            .attr("transform", rectTransform)
            .attr("height", 24)
            .attr("width", function(d) {
            return (x(d.endDate) - x(d.startDate));
            });
        svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0, " + (height - margin.top - margin.bottom) + ")")
            .transition()
            .call(xAxis);
        svg.append("g").attr("class", "y axis").transition().call(yAxis);

        //drawing legend
        function onlyUnique(value, index, self) {
            return self.indexOf(value) === index;
        }
        // select the svg area
        let SVG = svg.append("g");
        // Add legend boxes.
        let size = 10;
        SVG.selectAll("mydots")
            .data(keys)
            .enter()
            .append("rect")
            .attr("x", width_abs - 100)
            // 100 is where the first dot appears. 25 is the distance between dots
            .attr("y", function(d,i){ return i*(size+5)})
            .attr("width", size)
            .attr("height", size)
            .style("fill", function(d){ return colorList[d]});

        // Add legend names
        SVG.selectAll("mylabels")
            .data(keys)
            .enter()
            .append("text")
            .attr("x", width_abs - 100 + size * 1.3)
            // 100 is where the first dot appears. 25 is the distance between dots
            .attr("y", function(d,i){ return i*(size+5) + (size/2)})
            .style("fill", function(d){ return colorList[d]})
            .text(function(d){ return "Job " +d})
            .attr("text-anchor", "left")
            .attr("font-weight", 700)
            .style("alignment-baseline", "middle");
    return gantt;
}

    gantt.redraw = function(tasks) {
        initAxis();
        let svg = d3.select("svg");
        let ganttChartGroup = svg.select(".gantt-chart");
        let rect = ganttChartGroup.selectAll("rect").data(tasks, keyFunction);
        rect.enter()
            .insert("rect",":first-child")
            .attr("rx", 5)
            .attr("ry", 5)
            .attr("class", function(d){
                if(taskStatus[d.status] == null){ return "bar";}
                return taskStatus[d.status];
            })
            .transition()
            .attr("y", 0)
            .attr("transform", rectTransform)
            .attr("height", function(d) { return y.range()[1]; })
            .attr("width", function(d) {
                return (x(d.endDate) - x(d.startDate));
            });
            rect.transition()
                .attr("transform", rectTransform)
                .attr("height", function(d) { return y.range()[1]; })
                .attr("width", function(d) {
                    return (x(d.endDate) - x(d.startDate));
                });
            rect.exit().remove();
            svg.select(".x").transition().call(xAxis);
            svg.select(".y").transition().call(yAxis);
        return gantt;
    };

    gantt.margin = function(value) {
        if (!arguments.length)
            return margin;
        margin = value;
        return gantt;
    };

    gantt.timeDomain = function(value) {
        if (!arguments.length)
            return [ timeDomainStart, timeDomainEnd ];
        timeDomainStart = +value[0];
        timeDomainEnd = +value[1];
        return gantt;
    };

    /**
     * @param value The value can be "fit" - the domain fits the data or
     *                "fixed" - fixed domain.
     */
    gantt.timeBound = function(value) {
      if(!arguments.length)
          return timeDomainEnd;
      timeDomainEnd = value;
      return  gantt;
    };

    gantt.timeDomainMode = function(value) {
        if (!arguments.length)
            return timeDomainMode;
        timeDomainMode = value;
        return gantt;
    };

    gantt.taskTypes = function(value) {
        if (!arguments.length)
            return taskTypes;
        taskTypes = value;
        return gantt;
    };

    gantt.taskStatus = function(value) {
        if (!arguments.length)
            return taskStatus;
        taskStatus = value;
        return gantt;
    };

    gantt.width = function(value) {
        if (!arguments.length)
            return width;
        width = +value;
        return gantt;
    };

    gantt.height = function(value) {
        if (!arguments.length)
            return height;
        height = +value;
        return gantt;
    };

    gantt.tickFormat = function(value) {
        if (!arguments.length)
          return tickFormat;
        tickFormat = value;
        return gantt;
    };

    return gantt;
};