<h1 align="center">
	<img
		width="300"
		alt="coding duck MX"
		src="https://raw.githubusercontent.com/CodingDuckmx/hello-world/master/codingduckMX_logo.jpeg?sanitize=true">
</h1>

<h3 align="center">
	Data Scientist // Mathematician
</h3>

<p align="center">
	<strong>
    <a href="https://twitter.com/CodingDuckmx">Twitter</a>
		•
		<a href="https://medium.com/@CodingDuckMx">Blog</a>
		•
		<a href="https://www.linkedin.com/in/jesus-caballero-medrano/">LinkedIn</a>
	</strong>
</p>

# Sauti-Africa-Market-Monitoring
Sauti Africa Market Monitoring

The sequence of the sripts are the following:

1.- To verify the connection with the database is working:
  
    v2_verify_conn.py

2.- We need to create the schema of the data base.
    This will also populate the basic tables, such as contries,
    currencies, sources.
  
    v2_create_schema_db.py

3.- We collect the data from the original database with:

    v2_aws_collect_data.py
    
    We need the help of v2_dictionaries_and_lists.py to correct
    some ypos or errors in the original database. If it can't be
    done, the logs will be dropped in the error_logs table.
  
4.- Some basic cleaning like dropping prices at zero, trying 
     to fix prices that seem to be missing a decimal point, or otherwise
     clearly typos are dropped, and splittin the info into retail and
     wholesale is being doing here:
     
     v2_split_bc_drop.py
     
5.- A table with basis stats and statistic for the AD Fuller stationarity 
    test are dropped in the stats table with:
  
    v2_data_stats.py
  
6.- A meaninful data analysis is done and dropped into their specific table
    by the code written by Jing Qian (https://github.com/qianjing2020/):
    
    v2_qc_tables.py
 
7.- Build the bands for the normal, stress, alert or crisis status of 
    the prices with the script:
    
    v2_bands_construction.py
    
  
    
    
