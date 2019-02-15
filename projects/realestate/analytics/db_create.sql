create table dim_agency(
  pk_id serial primary key,
  sk_id varchar(256) null,
  agency_name varchar(256) null,
  agent_name varchar(256) null,
  agent_email varchar(256) null,
  agent_phonenumber varchar(256) null
);

create table dim_room(
  pk_id serial primary key,
  num_sleepingrooms int null,
  num_livingrooms int null,
  num_bathrooms int null,
  num_toilets int null,
  living_space int null
);

create table dim_address(
  pk_id serial primary key,
  city varchar(256) null,
  district varchar(256) null,
  street varchar(256) null,
  house_number varchar(256) null,
  latitude numeric(16,13) null,
  longitude numeric(16,13) null
);

create table dim_advertisement(
  pk_id serial primary key,
  state varchar(256) null,
  creation_date datetime null,
  last_update_date datetime null,
  has_pictures boolean null
);

create table dim_living_details(
  pk_id serial primary key,
  pets_allowed boolean null,
  assisted_living boolean null,
  construction_year int null,
  last_refurbishment int null
);

create table fact_flat(
  pk_id serial primary key,
  fk_agency BIGINT UNSIGNED NOT NULL,
  fk_room BIGINT UNSIGNED NOT NULL,
  fk_address BIGINT UNSIGNED NOT NULL,
  fk_advertisement BIGINT UNSIGNED NOT NULL,
  fk_living_details BIGINT UNSIGNED NOT NULL,
  base_rent numeric(11,2) null,
  service_charge numeric(11,2) null,
  total_rent numeric(11,2) null,
  foreign key (fk_agency) references dim_agency(pk_id),
  foreign key (fk_room) references dim_room(pk_id),
  foreign key (fk_address) references dim_address(pk_id),
  foreign key (fk_advertisement) references dim_advertisement(pk_id),
  foreign key (fk_living_details) references dim_living_details(pk_id)
);

create table event(
	insert_ts timestamp, 
	event_head varchar(256), 
	event text
);
