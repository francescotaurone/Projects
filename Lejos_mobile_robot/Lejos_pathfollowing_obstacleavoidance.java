package HelloWorld;

import javax.microedition.location.Orientation;

import lejos.nxt.Button; 
import lejos.nxt.LCD; 
import lejos.nxt.LightSensor; 
import lejos.nxt.MotorPort; 
import lejos.nxt.NXTMotor; 
import lejos.nxt.SensorPort; 
import lejos.nxt.Sound;
import lejos.nxt.UltrasonicSensor;
import lejos.nxt.NXTRegulatedMotor;

import java.io.File;
import java.lang.Math;

import lejos.nxt.Button; 
import lejos.nxt.LCD; 
import lejos.nxt.Motor;
import lejos.nxt.MotorPort; 
import lejos.nxt.NXTMotor; 
import lejos.nxt.SensorPort; 
import lejos.nxt.UltrasonicSensor;
import lejos.nxt.NXTRegulatedMotor;
import lejos.robotics.RegulatedMotor;
import lejos.robotics.navigation.DifferentialPilot;
import lejos.robotics.navigation.RotateMoveController;
import lejos.util.PilotProps;
import lejos.robotics.localization.OdometryPoseProvider;
import lejos.robotics.localization.PoseProvider;

public class finalProject {

	public static void main (String[] aArg) throws Exception{ 

		//CALIBRATION VARIABLES
		int line_color, floor_color;
		int val = 0, totalVal = 0, count = 0;

		//CONTROLLER VARIABLES
		NXTMotor mR = new NXTMotor(MotorPort.B); 
		NXTMotor mL = new NXTMotor(MotorPort.C); 
		UltrasonicSensor sonic = new UltrasonicSensor(SensorPort.S4);
		final LightSensor light = new LightSensor(SensorPort.S1); 
		NXTRegulatedMotor  mU = new NXTRegulatedMotor(MotorPort.A);
		
		//FOR OBJAVOIDANCE
		PilotProps pp = new PilotProps();
		pp.loadPersistentValues();
		float wheelDiameter = Float.parseFloat(pp.getProperty(PilotProps.KEY_WHEELDIAMETER, "5.8"));
		float trackWidth = Float.parseFloat(pp.getProperty(PilotProps.KEY_TRACKWIDTH, "16.0"));
		RegulatedMotor mRobj = PilotProps.getMotor(pp.getProperty(PilotProps.KEY_LEFTMOTOR, "B"));
		RegulatedMotor mLobj = PilotProps.getMotor(pp.getProperty(PilotProps.KEY_RIGHTMOTOR, "C"));
		boolean reverse = Boolean.parseBoolean(pp.getProperty(PilotProps.KEY_REVERSE,"false"));
		final DifferentialPilot pilot = new DifferentialPilot(wheelDiameter, trackWidth, mLobj, mRobj, reverse);
		PoseProvider pose = new OdometryPoseProvider(pilot);
		
		//DISTANCE VARIABLES
		double Kdistance = 3000;
		int flagDistance = 1;
//		double ksonicAngle = 0.03;
		double korientation  = 1;
		int sonicAngle = 0;
		double robotAngle = 0;
		int wallDistance = 0;
		int exitWallDistance = 20;
		int distanceThreshold = 50;
		double rotAngle = 0.0;
		int exitPower = 20;
		int lineCounter =0;

		int color;
		int colorBefore;
		int colorAfter;
		int colorErrorBefore = 0;
		int colorErrorAfter = 0;
		double LTurn; 
		double RTurn; 
		int threshold; 
		int nominalPower;
		String message;
		

		//CALIBRATION STEP
		
		Thread.sleep(1000); //pause for 1 second 
		light.setFloodlight(true); //turn LED floodlight on 

		// take a white calibration measurement 
		LCD.clear(); 
		LCD.drawString("Put it on the line", 0, 0);
		LCD.drawString("and press orange button", 0, 1);
		Button.ENTER.waitForPressAndRelease(); 

		while (count < 100){
			val = light.readNormalizedValue();
			totalVal = totalVal + val;          
			count++;
			Thread.sleep(10);
		}

		line_color = totalVal/count;
		light.setHigh(line_color); //set as high value 
		Sound.beep(); //beep once

		// take a black calibration measurement 
		LCD.drawString("FLOOR section", 0, 2); 
		LCD.drawString("LC value: " + line_color, 0, 6); 
		Button.ENTER.waitForPressAndRelease();

		val = 0; 
		totalVal = 0; 
		count = 0; 

		while (count < 100){
			val = light.readNormalizedValue();
			totalVal = totalVal + val;          
			count++;
			Thread.sleep(10);
		}

		floor_color = totalVal/count;
		light.setLow(floor_color); //set as low value 
		Sound.twoBeeps(); //beep twice 

		LCD.clear(); 
		LCD.drawString("Calibration done", 0, 0); 
		LCD.drawString("LC value: " + line_color, 0, 6); 
		LCD.drawString("FC value: " + floor_color, 0, 7); 
		LCD.refresh(); 
		Thread.sleep(5000); //pause for 5 seconds 
		Sound.buzz(); //buzz once

		//		EXIT THE SQUARE
		
		exitPower = 60;
		mR.setPower(new Double(exitPower).intValue()); 
		mL.setPower(new Double(exitPower).intValue()); 
		mR.forward(); 
		mL.forward();
		while(light.readValue()<80){
			;
		}
		while(light.readValue()>50){
			;
		}
		exitPower = 30;


		//		FOLLOW THE PATH

		
		double kp_neg = 3.2, kp_pos = 3.2;

		int error = 0; 

		double correction = 0;
		threshold = 50;

		while(!Button.ENTER.isDown()){
			while(sonic.getDistance()>exitWallDistance){
				color = light.readValue();

				LCD.clear();
				LCD.drawString("Color:" + color, 0, 2);

				error = color - threshold;
				nominalPower = 100;
				//
				//			HERE WE USE A DIFFERENT KP FOR LEFT OR RIGHT
				//			correction = kp * error;
				//			
				//			right_correction = kp_right * error;
				//			left_correction = kp_left * error;
				//			
				//
				//			RTurn = nominalPower - right_correction;
				//			LTurn = nominalPower + left_correction;

				//			HERE WE CHECK ON THE POS OR NEG ERROR

				if(error>=0)
					correction = kp_pos * error;
				else 
					correction = kp_neg * error;
				RTurn = nominalPower - correction;
				LTurn = nominalPower + correction;

				//message = "bT=" + new Double(RTurn).intValue() + " cT=" + new Double(LTurn).intValue();

				message = "pow= " + (new Double(RTurn).intValue() + new Double(LTurn).intValue());

				LCD.drawString(message, 0, 6, false); 
				mR.setPower(new Double(RTurn).intValue()); 
				mR.forward(); 
				mL.setPower(new Double(LTurn).intValue()); 
				mL.forward(); 

			}
			mR.stop();
			mL.stop();
			//			TURN TO BEGIN OBSTACLE AVOIDANCE

//			mR.setPower(new Double(exitPower).intValue()); 
//			mL.setPower(new Double(exitPower).intValue()); 
//			mR.forward(); 
//			mL.backward();
//			Thread.sleep(1000); //pause for 1 seconds 
//			mR.forward(); 
//			mL.forward();
			mR.setPower(new Double(exitPower).intValue()); 
			mL.setPower(new Double(exitPower).intValue()); 
			mR.forward(); 
			mL.backward();
			Thread.sleep(1500);
			mR.forward(); 
			mL.forward();
			while(error<0){
				color = light.readValue();

				LCD.clear();
				LCD.drawString("Color:" + color, 0, 2);

				error = color - threshold;
			}
			//OBSTACLE AVOIDANCE
			
			//PROPORTIONAL LAW
			sonic.continuous();
			mU.resetTachoCount();

			mU.setSpeed(200);
			mU.forward();

			while(lineCounter<3){
				colorBefore = light.readValue();
				
				
				
				sonicAngle = mU.getTachoCount();

				if(sonic.getDistance() < distanceThreshold){
					flagDistance = 1;
					mU.setSpeed(30);
					exitPower=15;
					korientation=0.3;
					if(sonic.getDistance() < distanceThreshold-20){
						exitPower=10;
						korientation=0.3;
					}
				}

				else{
					mU.setSpeed(200);
					flagDistance = 0;
					exitPower=20;
					korientation=1;
				}
				wallDistance = sonic.getDistance();
				robotAngle = pose.getPose().getHeading();
				//mU.setSpeed((int)(100+5*(double)wallDistance));

				LCD.clearDisplay();
				
				LCD.drawString("sonicAng:"+sonicAngle, 0, 5, false);
				LCD.drawString("robotAng:"+robotAngle, 0, 4, false);
				LCD.drawString("rotAngle"+rotAngle,0, 1, false);

				LCD.drawString("Posa:"+pose.getPose().getHeading(), 0, 6, false);
				LCD.drawString("Line#:"+lineCounter, 0, 3, false);

				pilot.setTravelSpeed(exitPower);
				pilot.forward();
				
				if(sonicAngle >=0){
					rotAngle = - korientation*robotAngle - flagDistance*(Kdistance*(1/((double)wallDistance)));
					if(sonicAngle>=45)
						mU.backward();
				}
				else{
					rotAngle = - korientation*robotAngle + flagDistance*(Kdistance*(1/((double)wallDistance)));
					if(sonicAngle<-45)
						mU.forward();
				}
				pilot.steer(rotAngle);
				colorAfter = light.readValue();
				colorErrorAfter = colorAfter - 50;
				colorErrorBefore = colorBefore -50;
				if(colorErrorBefore>0 && colorErrorAfter<0 )
					lineCounter++;
			} 
			pilot.setTravelSpeed(0);
			mU.stop();
//			File file= new File("StarWarsTheme.rbt");
//			Sound.playSample(file);
			
			Sound.buzz();
			Sound.beepSequenceUp();		
			Sound.beepSequenceUp();	
			Sound.beepSequenceUp();	

		} 

	}
}
