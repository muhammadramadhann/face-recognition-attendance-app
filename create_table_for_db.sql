CREATE TABLE `img_dataset` (
  `img_id` int(11) NOT NULL,
  `img_person` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
 
CREATE TABLE `prs_mstr` (
  `prs_nbr` varchar(10) NOT NULL,
  `prs_name` varchar(50) NOT NULL,
  `prs_class` varchar(30) NOT NULL,
  `prs_active` varchar(1) NOT NULL DEFAULT 'Y',
  `prs_added` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
 
ALTER TABLE `img_dataset`
  ADD PRIMARY KEY (`img_id`);
 
ALTER TABLE `prs_mstr`
  ADD PRIMARY KEY (`prs_nbr`);

CREATE TABLE `accs_hist` (
  `accs_id` int(11) NOT NULL AUTO_INCREMENT,
  `accs_date` date NOT NULL,
  `accs_prsn` varchar(10) NOT NULL,
  `accs_added` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`accs_id`),
  KEY `accs_date` (`accs_date`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;