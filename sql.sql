/*
SQLyog Community Edition- MySQL GUI v8.03 
MySQL - 5.6.12-log : Database - 22_lbs_sign
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`22_lbs_sign` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `22_lbs_sign`;

/*Table structure for table `alphabets` */

DROP TABLE IF EXISTS `alphabets`;

CREATE TABLE `alphabets` (
  `aid` int(11) NOT NULL AUTO_INCREMENT,
  `aname` varchar(20) DEFAULT NULL,
  `photo` varchar(200) DEFAULT NULL,
  PRIMARY KEY (`aid`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=latin1;

/*Data for the table `alphabets` */

insert  into `alphabets`(`aid`,`aname`,`photo`) values (2,'a','/static/alphabets/20230126185203.jpg');

/*Table structure for table `complaints` */

DROP TABLE IF EXISTS `complaints`;

CREATE TABLE `complaints` (
  `Complaint_id` int(11) NOT NULL AUTO_INCREMENT,
  `Complaints` varchar(100) DEFAULT NULL,
  `Complaint_date` varchar(10) DEFAULT NULL,
  `Reply` varchar(100) DEFAULT 'pending',
  `Reply_date` varchar(10) DEFAULT '0000-00-00',
  `User_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`Complaint_id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=latin1;

/*Data for the table `complaints` */

insert  into `complaints`(`Complaint_id`,`Complaints`,`Complaint_date`,`Reply`,`Reply_date`,`User_id`) values (1,'xxx',NULL,'xxxxx','2022-12-28',4),(2,'ccc',NULL,'kk','2023-01-26',4),(3,'bmmm','2023-01-15','pending','0000-00-00',4),(4,'hdfghdgd','2023-01-26','pending','0000-00-00',2);

/*Table structure for table `login` */

DROP TABLE IF EXISTS `login`;

CREATE TABLE `login` (
  `login_id` int(100) NOT NULL AUTO_INCREMENT,
  `user_name` varchar(100) DEFAULT NULL,
  `password` varchar(100) DEFAULT NULL,
  `user_type` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`login_id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=latin1;

/*Data for the table `login` */

insert  into `login`(`login_id`,`user_name`,`password`,`user_type`) values (1,'admin@gmail.com','a','admin'),(2,'u@gmail.com','a','user'),(3,'xxx@gmail.com','Admin@12','user'),(4,'xxx@gmail.com','Admin@12','user'),(6,'h5@gmail.com','1111111','user');

/*Table structure for table `suggestion` */

DROP TABLE IF EXISTS `suggestion`;

CREATE TABLE `suggestion` (
  `sid` int(11) NOT NULL AUTO_INCREMENT,
  `uid` int(11) DEFAULT NULL,
  `suggestion` mediumtext,
  `sdate` date DEFAULT NULL,
  PRIMARY KEY (`sid`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=latin1;

/*Data for the table `suggestion` */

insert  into `suggestion`(`sid`,`uid`,`suggestion`,`sdate`) values (1,3,NULL,NULL),(2,4,NULL,NULL),(3,5,NULL,NULL),(4,2,'sdjkhsdkhd','2023-01-26'),(5,2,'ckdkzfjd','2023-01-26'),(6,2,'zjxjsdh','2023-01-26');

/*Table structure for table `test` */

DROP TABLE IF EXISTS `test`;

CREATE TABLE `test` (
  `test_id` int(11) NOT NULL AUTO_INCREMENT,
  `uid` int(11) DEFAULT NULL,
  `total_mark` int(11) DEFAULT NULL,
  `date` date DEFAULT NULL,
  PRIMARY KEY (`test_id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=latin1;

/*Data for the table `test` */

insert  into `test`(`test_id`,`uid`,`total_mark`,`date`) values (1,2,34,'2023-01-01'),(2,2,1,'2022-12-31'),(3,2,33,'2023-01-03'),(4,2,9,'2023-01-22');

/*Table structure for table `user` */

DROP TABLE IF EXISTS `user`;

CREATE TABLE `user` (
  `user_id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(200) DEFAULT NULL,
  `email` varchar(200) DEFAULT NULL,
  `phonenumber` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=latin1;

/*Data for the table `user` */

insert  into `user`(`user_id`,`name`,`email`,`phonenumber`) values (2,'Neethi','xxx@gmail.com',9876543217),(3,'Neethi','xxx@gmail.com',9876543217),(4,'Neethi','xxx@gmail.com',9876543217),(6,'Anshika','h5@gmail.com',9876543678);

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
