with time_z as(
	SELECT DISTINCT
	    cast(cfd.Coid as VARCHAR(5)) AS coid,
	    cfd.Time_Zone_Name AS time_zone_name,
	    CASE
	        WHEN cfd.Time_Zone_Name = 'America Eastern'  THEN 'US/Eastern'
	    WHEN cfd.Time_Zone_Name = 'America Central'  THEN 'US/Central'
	    WHEN cfd.Time_Zone_Name = 'America Mountain' THEN 'US/Mountain'
	    WHEN cfd.Time_Zone_Name = 'America Pacific'  THEN 'US/Pacific'
	    WHEN cfd.Time_Zone_Name = 'America Alaska'   THEN 'US/Alaska'
	        ELSE NULL
	    END AS time_zone_offset_value
	FROM EDWCDM_Views.CDM_Facility_Detail cfd
	INNER JOIN EDW_DIM_Views.Dim_Organization coid_tb
	ON coid_tb.Coid = cfd.Coid
	WHERE 1=1
	AND cfd.Coid = coid_tb.Coid
	and coid_tb.Coid IS NOT NULL
),
org_sids as(
	select 
	DISTINCT
	org_sid,
	coid
	from EDW_DIM_VIEWS.dim_organization	
)
select
org_sid as SERIES_COL, 
time_zone_name,
time_zone_offset_value
from time_z 
left join org_sids
on org_sids.coid=time_z.coid
order by org_sids.coid
